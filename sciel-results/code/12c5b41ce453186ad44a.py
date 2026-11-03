"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Hexagonal lattice packing
    # Centers are at (x, y)
    # Lattice basis: u=(2,0), v=(1,sqrt(3))
    # Points:
    # Center: (0,0)
    # Ring 1: (2,0), (-2,0), (1,sqrt(3)), (-1,sqrt(3)), (1,-sqrt(3)), (-1,-sqrt(3))
    # Ring 2 additions: (3,sqrt(3)), (-3,sqrt(3)), (3,-sqrt(3)), (-3,-sqrt(3))
    
    sqrt3 = math.sqrt(3)
    
    inner = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [1.0, sqrt3, 0.0],
        [-1.0, sqrt3, 0.0],
        [1.0, -sqrt3, 0.0],
        [-1.0, -sqrt3, 0.0],
        [3.0, sqrt3, 0.0],
        [-3.0, sqrt3, 0.0],
        [3.0, -sqrt3, 0.0],
        [-3.0, -sqrt3, 0.0]
    ]
    
    # Outer hexagon
    # Side length 5.0 fits this configuration
    center = (0.0, 0.0)
    side = 5.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
