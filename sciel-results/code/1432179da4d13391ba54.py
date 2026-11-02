"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Compact honeycomb packing of 11 unit hexagons
    # Orientation: Pointy-topped (vertices at 90, 150, etc)
    # Nearest neighbor distance: sqrt(3)
    
    sqrt3 = math.sqrt(3)
    inner = []
    
    # Row 0 (y=0): 5 hexagons
    for i in range(-2, 3):
        inner.append([i * sqrt3, 0.0, 0.0])
        
    # Row 1 (y=1.5): 3 hexagons
    for i in range(-1, 2):
        # x coordinates are shifted by sqrt(3)/2 relative to row 0
        # Row 0 x: -2, -1, 0, 1, 2 (scaled by sqrt3)
        # Row 1 x: -1.5, -0.5, 0.5 (scaled by sqrt3)
        x = (i - 0.5) * sqrt3
        inner.append([x, 1.5, 0.0])
        
    # Row -1 (y=-1.5): 3 hexagons
    for i in range(-1, 2):
        x = (i - 0.5) * sqrt3
        inner.append([x, -1.5, 0.0])
        
    # Outer hexagon parameters
    # Bounding box of inner hexagons:
    # X: -2.5*sqrt3 to 2.5*sqrt3 (Width = 5*sqrt3)
    # Y: -2.5 to 2.5 (Height = 5)
    # Optimal enclosing flat-topped hexagon side S:
    # Constraint: 1.5*S - 2.5 >= 2.5*sqrt3
    # S >= (2.5*sqrt3 + 2.5) / 1.5
    S = (2.5 * sqrt3 + 2.5) / 1.5
    # Add small margin
    S += 0.01
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
