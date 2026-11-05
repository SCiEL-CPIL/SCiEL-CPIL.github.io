import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define a compact cluster of 12 unit hexagons.
    # We use a hexagonal lattice arrangement.
    # Lattice vectors:
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    # 
    # We select 12 centers:
    # 1 center at (0,0)
    # 6 neighbors at distance sqrt(3)
    # 5 more at distance 3 (from the 6 available at distance 3, we pick 5)
    #
    # To ensure validity and symmetry, we calculate the required outer side length.
    # For this specific cluster, the maximum distance from the origin to a center is 3.
    # With unit hexagons aligned to the axes, the maximum extent is 3 + 1 = 4.
    # Thus, an outer hexagon of side length 4.0 is sufficient.
    
    sqrt3 = math.sqrt(3)
    
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (-sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (sqrt3/2, -1.5),
        (-sqrt3/2, -1.5),
        # The next shell has 6 points at distance 3.
        # (3*sqrt3/2, 1.5), (-3*sqrt3/2, 1.5), (3*sqrt3/2, -1.5), (-3*sqrt3/2, -1.5), (0, 3), (0, -3)
        # We pick 5 of them. To maintain balance, we might pick 5, but any 5 will work for L=4.
        # Let's pick:
        (3*sqrt3/2, 1.5),
        (-3*sqrt3/2, 1.5),
        (3*sqrt3/2, -1.5),
        (-3*sqrt3/2, -1.5),
        (0.0, 3.0)
    ]
    
    # Verify we have 12
    assert len(centers) == N
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
