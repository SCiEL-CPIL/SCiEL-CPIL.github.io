import math

N = 12

def construct():
    # 1 center hexagon
    inner = [[0.0, 0.0, 0.0]]
    
    # 6 inner ring hexagons
    # Distance sqrt(3), angles 30, 90, ...
    for i in range(6):
        angle_deg = 30 + 60 * i
        angle_rad = math.radians(angle_deg)
        x = math.sqrt(3) * math.cos(angle_rad)
        y = math.sqrt(3) * math.sin(angle_rad)
        # Orientation: keep same as center (0 degrees) for simplicity?
        # If center is 0, sides are at 30, 90...
        # Neighbor at 30 deg angle. Side facing center is at 210 deg.
        # If neighbor is 0 deg, its side at 210 is parallel to center's side at 30.
        # So 0 is correct.
        inner.append([x, y, 0.0])
        
    # 5 outer ring hexagons
    # Distance 3.0, angles 0, 60, 120, 180, 240
    for i in range(5):
        angle_deg = 0 + 60 * i
        angle_rad = math.radians(angle_deg)
        x = 3.0 * math.cos(angle_rad)
        y = 3.0 * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
