import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 3 rows, 4-3-4 pattern
    # Centered at (0,0)
    # y offsets: -sqrt(3), 0, sqrt(3)
    # x offsets for y=-sqrt(3): -3, -1, 1, 3
    # x offsets for y=0: -2, 0, 2
    # x offsets for y=sqrt(3): -3, -1, 1, 3
    
    inner = []
    s = math.sqrt(3)
    
    # Row 1 (bottom)
    for x in [-3, -1, 1, 3]:
        inner.append([x, -s, 0.0])
        
    # Row 2 (middle)
    for x in [-2, 0, 2]:
        inner.append([x, 0.0, 0.0])
        
    # Row 3 (top)
    for x in [-3, -1, 1, 3]:
        inner.append([x, s, 0.0])
        
    # Apply gap factor
    gap = 1.0001
    for i in range(len(inner)):
        inner[i][0] *= gap
        inner[i][1] *= gap
        
    # Outer hexagon
    # For 4-3-4 flat-top packing, min side L = 4*gap + 1.5
    # Adding small epsilon for numerical safety
    L = 4.0 * gap + 1.5 + 1e-5
    
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
