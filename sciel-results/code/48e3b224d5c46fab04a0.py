import math

N = 12

def construct():
    # 7-hex cluster: center + 6 neighbors
    # 5 extra hexes on 5 of the 6 sides
    
    centers = []
    
    # Center
    centers.append((0.0, 0.0))
    
    # 6 neighbors
    for i in range(6):
        angle = math.radians(60 * i)
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
        
    # 5 extra hexes at distance 2*sqrt(3) from center
    # Angles: 0, 60, 120, 180, 240 (missing 300)
    for i in range(5):
        angle = math.radians(60 * i)
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    side_length = 2 * math.sqrt(3) + 1.0
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side_length, angle
