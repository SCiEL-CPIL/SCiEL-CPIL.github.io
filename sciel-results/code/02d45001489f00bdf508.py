import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Scaled coordinates where minimum distance is 1
    # 7-point flower
    pts = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, math.sqrt(3)/2.0),
        (-0.5, math.sqrt(3)/2.0),
        (0.5, -math.sqrt(3)/2.0),
        (-0.5, -math.sqrt(3)/2.0)
    ]
    # 4 additional points to make 11
    # Extending the arms
    pts.append((2.0, 0.0))
    pts.append((-2.0, 0.0))
    pts.append((1.0, math.sqrt(3)))
    pts.append((-1.0, math.sqrt(3)))
    
    # Scale to real coordinates
    # Distance between centers must be >= sqrt(3)
    scale = math.sqrt(3)
    
    inner = []
    for x, y in pts:
        inner.append([x * scale, y * scale, 0.0])
        
    # Outer hexagon side
    # Container side in scaled units S = 2
    # Real side L = 1 + S * sqrt(3)
    L = 1.0 + 2.0 * math.sqrt(3)
    
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
