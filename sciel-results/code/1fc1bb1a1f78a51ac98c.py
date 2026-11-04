import math

N = 12

def construct():
    # Side length of outer hexagon
    # S = 1 + 2 * sqrt(3)
    S = 1.0 + 2.0 * math.sqrt(3.0)
    
    # Inner hexagon centers
    # All aligned with angle 0
    centers = []
    
    # Ring 0
    centers.append((0.0, 0.0))
    
    # Ring 1 (6 points)
    r1 = math.sqrt(3.0)
    for i in range(6):
        angle = math.radians(60 * i)
        x = r1 * math.cos(angle)
        y = r1 * math.sin(angle)
        centers.append((x, y))
        
    # Ring 2 (Pick 5 points from the 6 at distance 2*sqrt(3))
    r2 = 2.0 * math.sqrt(3.0)
    for i in range(5):
        angle = math.radians(60 * i)
        x = r2 * math.cos(angle)
        y = r2 * math.sin(angle)
        centers.append((x, y))
        
    # Format inner list
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
