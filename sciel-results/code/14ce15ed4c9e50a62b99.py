import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate the 11 positions
    # Center
    positions = [(0.0, 0.0)]
    
    # Ring 1 (6 points)
    # Axial (q, r) -> Cartesian (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    ring1_axial = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for q, r in ring1_axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
        
    # Ring 2 Side Points (6 points, dist 3)
    # We only need 4 more to reach 11.
    # Axial coords for side points:
    ring2_side_axial = [(1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)]
    
    # Take first 4
    for i in range(4):
        q, r = ring2_side_axial[i]
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
        
    # Format inner hexagons: [x, y, angle]
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
