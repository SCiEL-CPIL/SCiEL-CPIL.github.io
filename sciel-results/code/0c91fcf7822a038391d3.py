import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Outer hexagon parameters
    outer_center = (0.0, 0.0)
    outer_side = 4.0
    outer_angle = 30.0  # Rotate so vertices align with the 2nd ring hexes at 30, 90, etc.

    # Inner hexagons
    inner = []
    
    # 1. Center
    inner.append([0.0, 0.0, 0.0])
    
    # 2-7. First ring (6 hexes) at distance sqrt(3)
    # Angles: 0, 60, 120, 180, 240, 300
    dist1 = math.sqrt(3)
    for i in range(6):
        angle_rad = math.radians(60 * i)
        x = dist1 * math.cos(angle_rad)
        y = dist1 * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    # 8-12. Second ring (5 hexes) at distance 3
    # We pick angles 30, 90, 150, 210, 270 (skipping 330)
    dist2 = 3.0
    angles_to_pick = [30, 90, 150, 210, 270]
    for angle_deg in angles_to_pick:
        angle_rad = math.radians(angle_deg)
        x = dist2 * math.cos(angle_rad)
        y = dist2 * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
