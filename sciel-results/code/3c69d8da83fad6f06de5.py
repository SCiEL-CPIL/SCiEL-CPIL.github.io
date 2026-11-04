import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate axial coordinates for a hexagon of radius 2 (side length 3 in hex units)
    # This cluster contains 19 hexagons. We need 11.
    # We will select a subset that is centrally located.
    
    coords = []
    # Radius 0
    coords.append((0, 0))
    # Radius 1 (6 hexagons)
    for i in range(6):
        # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        dq, dr = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)][i]
        coords.append((dq, dr))
    # Radius 2 (12 hexagons)
    # We need 11 - 7 = 4 more.
    # Let's pick 4 from radius 2 to maintain some symmetry or just take the first 4.
    # To minimize the bounding box, we might want to avoid corners if we were free, 
    # but in a hex grid, all radius 2 points have max coordinate 2.
    # So any 4 will result in the same bounding box size S=5.
    # Let's pick them symmetrically: (2,0), (-2,0), (0,2), (0,-2) in axial?
    # Axial (2,0) -> x=2sqrt(3), y=0
    # Axial (-2,0) -> x=-2sqrt(3), y=0
    # Axial (0,2) -> x=sqrt(3), y=3
    # Axial (0,-2) -> x=-sqrt(3), y=-3
    # These are on the axes.
    
    extra_coords = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    coords.extend(extra_coords)
    
    # Convert to Cartesian
    inner = []
    for q, r in coords[:N]:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # Outer hexagon parameters
    # Side length S such that it contains the cluster.
    # Max coordinate in axial is 2.
    # Distance from center to flat side of outer hexagon is S * sqrt(3) / 2.
    # Distance from center to flat side of inner hexagon at axial distance D is D * sqrt(3) + sqrt(3)/2.
    # Here max D is 2. So dist = 2.5 * sqrt(3).
    # S * sqrt(3) / 2 >= 2.5 * sqrt(3) => S >= 5.
    side = 5.0
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
