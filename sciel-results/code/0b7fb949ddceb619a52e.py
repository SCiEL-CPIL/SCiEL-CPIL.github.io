import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Coordinates of the 11 inner hexagon centers in a hexagonal lattice
    # Axial coordinates (q, r) -> Cartesian (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    sqrt3 = math.sqrt(3)
    
    # Define the set of 11 centers
    # Center
    centers = []
    centers.append((0, 0))
    
    # Ring 1 (6 neighbors)
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for q, r in ring1:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Ring 2 (4 additional neighbors to make 11 total)
    # We choose (2,0), (0,2), (-2,0), (0,-2) for a cross shape
    ring2 = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    for q, r in ring2:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Prepare inner list: [x, y, angle_degrees]
    # All inner hexagons are aligned with the outer one (angle 0)
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    # Outer hexagon parameters
    # Side length S = 2*sqrt(3) + 1
    side = 2 * sqrt3 + 1
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
