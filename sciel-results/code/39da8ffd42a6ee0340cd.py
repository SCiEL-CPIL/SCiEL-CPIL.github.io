import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Calculate optimal side length
    # S = 2*sqrt(3) + 1
    S = 2 * math.sqrt(3) + 1.0
    
    inner = []
    
    # Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    dist1 = math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        x = dist1 * math.cos(angle)
        y = dist1 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Ring 2: We need 4 more. 
    # Pick the 4 "corner" positions (aligned with axes) at distance 2*sqrt(3)
    # These are at angles 0, 60, 120, 180
    dist2 = 2 * math.sqrt(3)
    for i in range(4):
        angle = math.radians(60 * i)
        x = dist2 * math.cos(angle)
        y = dist2 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Total 1 + 6 + 4 = 11 hexagons
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
