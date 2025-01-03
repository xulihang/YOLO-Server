#!/usr/bin/env python3

import os
import time
import datetime
from PIL import Image
from io import BytesIO
from bottle import BaseRequest, route, run, template, request, static_file
import json
import base64
from ultralytics import YOLO
BaseRequest.MEMFILE_MAX = 1024 * 1024 * 10 # (or whatever you want)

@route('/detect', method='POST')
def detect():
    print("detect")
    json = request.json
    image = json["image"]
    bytes_decoded = base64.b64decode(image)
    net_img = Image.open(BytesIO(bytes_decoded))
    
    prediction = model.predict(source=net_img, conf=0.5)[0]
    ret = {}
    results = []
    if prediction.boxes != None:
        for box in prediction.boxes:
            cls = int(box.cls)
            class_name = model.names[cls]
            x_center,y_center,w,h = box.xywh[0].tolist()
            x = x_center - 0.5*w
            y = y_center - 0.5*h
            location = {
              "left":x,
              "top":y,
              "width":w,
              "height":h,
              "className":class_name
            }
            results.append({"location":location})
            
    elif prediction.obb != None:
        for obb in prediction.obb:
            x_center,y_center,w,h,r = obb.xywhr[0].tolist()
            x = x_center - 0.5*w
            y = y_center - 0.5*h
            location = {
              "left":x,
              "top":y,
              "width":w,
              "height":h,
              "rotation":r,
              "className":class_name
            }
            results.append({"location":location})
    ret["results"] = results
    return ret


@route('/<filepath:path>')
def server_static(filepath):
    return static_file(filepath, root='www')
    
model = YOLO("yolo11n.pt")
print(model.names)

run(server="paste",host='127.0.0.1', port=8085)     