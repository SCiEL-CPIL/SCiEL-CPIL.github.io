"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Using a 4x3 grid of Pointy-Top hexagons
    # Pointy-Top: Vertex at top (90 deg). Horizontal spacing sqrt(3), Vertical spacing 1.5.
    # Grid: 4 columns, 3 rows.
    
    sqrt3 = math.sqrt(3)
    
    # X coordinates for 4 columns centered at 0
    # Spacing is sqrt(3)
    # Centers: -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    x_coords = [-1.5 * sqrt3, -0.5 * sqrt3, 0.5 * sqrt3, 1.5 * sqrt3]
    
    # Y coordinates for 3 rows centered at 0
    # Spacing is 1.5
    # Centers: -1.5, 0, 1.5
    y_coords = [-1.5, 0.0, 1.5]
    
    inner = []
    for y in y_coords:
        for x in x_coords:
            # Angle 90 degrees for Pointy-Top hexagons (vertex at top)
            inner.append([x, y, 90.0])
            
    # Outer hexagon
    # Width of grid: 4 * sqrt(3) = 6.928
    # Height of grid: 5.0
    # Align width with long diagonal of hexagon (2S)
    # 2S >= 4 * sqrt(3) => S >= 2 * sqrt(3) = 3.464
    side = 2.0 * sqrt3
    
    # Check height constraint: sqrt(3) * S >= 5.0
    # sqrt(3) * 2 * sqrt(3) = 6.0 >= 5.0. OK.
    
    center = (0.0, 0.0)
    angle = 0.0 # Outer hexagon vertices on X-axis
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
