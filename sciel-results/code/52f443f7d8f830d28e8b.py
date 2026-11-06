import numpy as np
import time

def construct():
    """
    Returns 30 distinct integer coordinate pairs in [-30, 30]^2.
    Strategy: 25 points on the x-axis (0,0) to (24,0) and 5 points on the y-axis (0,1) to (0,5).
    This maximizes the multiplicity of doubled area 4.
    """
    points = []
    # 25 points on the x-axis
    for i in range(25):
        points.append((i, 0))
    # 5 points on the y-axis
    for j in range(1, 6):
        points.append((0, j))
        
    return points
