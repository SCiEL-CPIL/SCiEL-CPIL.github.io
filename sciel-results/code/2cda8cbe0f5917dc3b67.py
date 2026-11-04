import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Axial coordinates for the 11 hexagons
    # Center
    coords = [(0, 0)]
    # Ring 1 (6 hexagons)
    coords.extend([
        (1, 0), (1, -1), (0, -1),
        (-1, 0), (-1, 1), (0, 1)
    ])
    # Ring 2 selection (4 hexagons)
    coords.extend([
        (1, 1), (-1, -1),
        (0, 2), (0, -2)
    ])
    
    inner = []
    for q, r in coords:
        # Convert axial to Cartesian
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    center = (0.0, 0.0)
    side = 4.0
    # Rotate outer hexagon by 30 degrees to align vertices with the extreme points
    angle = 30.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
