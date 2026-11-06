import random
import numpy as np
from collections import Counter
from itertools import combinations

random.seed(42)

def construct():
    """
    Construct 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area (absolute determinant).
    
    Approach: Use two parallel lines (y=0 and y=1) with 15 points each, 
    spaced by 1 unit in x. This creates a high multiplicity of triangles 
    with doubled area 1 (base 1, height 1).
    """
    points = []
    # 15 points on y=0, x from -14 to 14
    for x in range(-14, 15):
        points.append((x, 0))
    # 15 points on y=1, x from -14 to 14
    for x in range(-14, 15):
        points.append((x, 1))
        
    return points
