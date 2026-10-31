import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    d = math.sqrt(3)
    h = 1.5
    
    # 4-3-4 pattern
    # Row 0: 4 points at y=0
    # Row 1: 3 points at y=h
    # Row 2: 4 points at y=2h
    
    centers = []
    
    # Row 0
    for x in [-1.5*d, -0.5*d, 0.5*d, 1.5*d]:
        centers.append([x, 0.0, 0.0])
        
    # Row 1
    for x in [-d, 0.0, d]:
        centers.append([x, h, 0.0])
        
    # Row 2
    for x in [-1.5*d, -0.5*d, 0.5*d, 1.5*d]:
        centers.append([x, 2*h, 0.0])
        
    # Outer hexagon
    # Centered at (0, h)
    # Side length 4.0
    # Angle 0.0
    
    return centers, (0.0, h), 4.0, 0.0
# EVOLVE-BLOCK-END
