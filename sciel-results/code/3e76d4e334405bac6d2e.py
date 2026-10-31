"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import math

# EVOLVE-BLOCK-START
def construct():
    # Packing 11 unit hexagons into a regular hexagon of side length 4.5
    # Arrangement: 3 in bottom row, 4 in middle row, 4 in top row.
    # Shifted down by 0.5 to fit within the corners.
    
    sqrt3 = math.sqrt(3)
    
    # Row 0 (Bottom): 3 hexagons
    y0 = -2.0
    x0s = [-sqrt3, 0.0, sqrt3]
    
    # Row 1 (Middle): 4 hexagons
    y1 = -0.5
    x1s = [-1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3]
    
    # Row 2 (Top): 4 hexagons
    y2 = 1.0
    x2s = [-sqrt3, 0.0, sqrt3, 2.0*sqrt3]
    
    inner = []
    for x in x0s:
        inner.append([x, y0, 0.0])
    for x in x1s:
        inner.append([x, y1, 0.0])
    for x in x2s:
        inner.append([x, y2, 0.0])
        
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.5
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
