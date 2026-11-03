import math

def construct():
    sqrt3 = math.sqrt(3)
    
    # Core 7
    inner = [
        [0.0, 0.0, 0.0],
        [sqrt3, 0.0, 0.0],
        [sqrt3/2, 1.5, 0.0],
        [-sqrt3/2, 1.5, 0.0],
        [-sqrt3, 0.0, 0.0],
        [-sqrt3/2, -1.5, 0.0],
        [sqrt3/2, -1.5, 0.0]
    ]
    
    # 4 Outward
    # 0 deg
    inner.append([2*sqrt3, 0.0, 0.0])
    # 60 deg
    inner.append([sqrt3, 3.0, 0.0])
    # 120 deg
    inner.append([-sqrt3, 3.0, 0.0])
    # 180 deg
    inner.append([-2*sqrt3, 0.0, 0.0])
    
    side = 2*sqrt3 + 1.0
    
    return inner, (0.0, 0.0), side, 0.0
