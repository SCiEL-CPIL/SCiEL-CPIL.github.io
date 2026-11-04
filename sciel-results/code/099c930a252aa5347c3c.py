import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Configuration: 3-4-3-2 hexagonal grid
    # Shifted to center vertically
    dy = math.sqrt(3) / 2
    row_y = [1.5 * dy, 0.5 * dy, -0.5 * dy, -1.5 * dy]
    
    inner = []
    
    # Row 0: 3 hexagons
    for x in [-2, 0, 2]:
        inner.append([x, row_y[0], 0.0])
        
    # Row 1: 4 hexagons
    for x in [-3, -1, 1, 3]:
        inner.append([x, row_y[1], 0.0])
        
    # Row 2: 3 hexagons
    for x in [-2, 0, 2]:
        inner.append([x, row_y[2], 0.0])
        
    # Row 3: 2 hexagons
    for x in [-1, 1]:
        inner.append([x, row_y[3], 0.0])
        
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.5
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
