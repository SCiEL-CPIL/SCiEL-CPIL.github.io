import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 7 hexagons in a flower pattern
    # Center
    inner = [[0.0, 0.0, 0.0]]
    
    # 6 neighbors
    r1 = math.sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3
        x = r1 * math.cos(angle)
        y = r1 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # 4 additional hexagons at radius 3, angles 30, 90, 150, 210 degrees
    r2 = 3.0
    angles_deg = [30, 90, 150, 210]
    for deg in angles_deg:
        angle = math.radians(deg)
        x = r2 * math.cos(angle)
        y = r2 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.0
    angle_outer = 0.0
    
    return inner, center, side, angle_outer
# EVOLVE-BLOCK-END
