import math

N = 11

def construct():
    sqrt3 = math.sqrt(3)
    # Define centers
    centers = [
        (0.0, 0.0),
        (1.5, sqrt3/2),
        (1.5, -sqrt3/2),
        (0.0, -sqrt3),
        (-1.5, -sqrt3/2),
        (-1.5, sqrt3/2),
        (0.0, sqrt3),
        (3.0, 0.0),
        (-3.0, 0.0),
        (1.5, 1.5*sqrt3),
        (-1.5, -1.5*sqrt3)
    ]
    
    # Format for return: list of [x, y, angle]
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
