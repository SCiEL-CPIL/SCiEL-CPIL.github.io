"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Optimal packing for N=12 is a 3x4 grid of hexagons.
    # Side length L ~ 4.1 is sufficient. We use 4.2 for safety.
    L = 4.2
    
    inner = []
    # Row 0 (y=0): 4 hexagons
    # Centers at x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    # Row 1 (y=1.5): 4 hexagons, shifted by +0.5*sqrt(3)
    # Centers at x = -1.0*sqrt(3), 0.0, 1.0*sqrt(3), 2.0*sqrt(3)
    # Row -1 (y=-1.5): 4 hexagons, shifted by -0.5*sqrt(3)
    # Centers at x = -2.0*sqrt(3), -1.0*sqrt(3), 0.0, 1.0*sqrt(3)
    
    sqrt3 = math.sqrt(3)
    
    # Row 0
    for i in range(4):
        x = (i - 1.5) * sqrt3
        y = 0.0
        inner.append([x, y, 0.0])
        
    # Row 1
    for i in range(4):
        x = (i - 1.0) * sqrt3
        y = 1.5
        inner.append([x, y, 0.0])
        
    # Row -1
    for i in range(4):
        x = (i - 2.0) * sqrt3
        y = -1.5
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
