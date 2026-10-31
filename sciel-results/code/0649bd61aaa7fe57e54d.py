"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place unit regular hexagons on a hexagonal lattice (triangular grid)
    inside a concentric regular hexagon container.
    
    Strategy:
    - Unit hexagons have circumradius 1, inradius sqrt(3)/2, width sqrt(3).
    - Place centers on a triangular lattice with spacing sqrt(3) (the width of a unit hex).
    - The lattice is aligned so that the natural hexagonal symmetry matches the container.
    - For a hexagonal arrangement of side length s (in lattice units), the number of hexagons
      is 3*s*(s-1)+1. We choose the largest s such that 3*s*(s-1)+1 <= N.
    - The outer container must contain all these hexagons. The farthest point from the center
      in the lattice arrangement determines the required container size.
    
    For a hexagonal cluster of side s (s rows in each direction), the vertices of the
    bounding hexagon in lattice coordinates are at distance (s-1) from center along
    the 6 lattice directions. The actual Euclidean distance from center to the farthest
    hexagon center is (s-1)*sqrt(3). But we also need to account for the hexagon's own
    radius: the farthest point of a unit hexagon from the cluster center is at distance
    (s-1)*sqrt(3) + 1 (circumradius of unit hex).
    
    The outer regular hexagon with side length L has inradius L*sqrt(3)/2 and circumradius L.
    To contain all points, we need the outer hexagon's inradius >= the maximum distance
    from center to any point of the inner hexagons, when measured perpendicular to the
    nearest side. Due to symmetry, the worst case is along the 6 axes of the hexagonal
    lattice.
    
    The distance from center to the farthest point along a lattice axis direction:
    - The farthest hexagon center is at distance d_c = (s-1)*sqrt(3) from origin.
    - The unit hexagon extends an additional 1 (its circumradius) in that direction.
    - So the required inradius of the outer hexagon is (s-1)*sqrt(3) + 1.
    - Inradius of outer hexagon with side L is L*sqrt(3)/2.
    - So L*sqrt(3)/2 >= (s-1)*sqrt(3) + 1
    - L >= 2*(s-1) + 2/sqrt(3)
    
    Let's verify: for s=1 (single hex), L >= 0 + 2/sqrt(3) = 2/sqrt(3) ≈ 1.1547.
    A single unit hexagon has inradius sqrt(3)/2 ≈ 0.866. The outer hexagon with side
    2/sqrt(3) has inradius (2/sqrt(3))*sqrt(3)/2 = 1. That's correct: the outer hexagon
    must have inradius at least 1 to contain the unit hexagon (whose circumradius is 1,
    but the minimum containing hexagon aligned the same way needs inradius = circumradius = 1).
    Wait, actually for a regular hexagon containing another regular hexagon of the same
    orientation, the containing hexagon's inradius must be >= the contained hexagon's
    circumradius. So inradius_outer >= 1, meaning L*sqrt(3)/2 >= 1, L >= 2/sqrt(3). Correct.
    """
    # Find largest s such that 3*s*(s-1) + 1 <= N
    s = 1
    while 3 * (s + 1) * s + 1 <= N:
        s += 1
    
    # Generate hexagonal lattice positions
    # Use axial coordinates: (q, r) where q + r <= s-1, q >= -(s-1), r >= -(s-1)
    # and q + r >= -(s-1)
    positions = []
    for q in range(-(s - 1), s):
        for r in range(max(-(s - 1), -q - (s - 1)), min(s - 1, -q + (s - 1)) + 1):
            if q + r >= -(s - 1) and q + r <= s - 1:
                positions.append((q, r))
    
    # Convert axial to Cartesian: x = sqrt(3)*(q + r/2), y = (3/2)*r
    sqrt3 = math.sqrt(3)
    centers = []
    for q, r in positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Take first N positions (or all if fewer)
    centers = centers[:N]
    
    # Compute required outer side length
    # The farthest distance from origin to any hexagon center:
    max_center_dist = max(math.sqrt(x**2 + y**2) for x, y in centers) if centers else 0.0
    
    # But we need the inradius requirement. The outer hexagon is aligned with the lattice.
    # The inradius must be >= max over all points of the dot product with each of the 6
    # normal directions. For a regular hexagon centered at origin with flat-top orientation
    # (or pointy-top), the 6 side normals are at angles 0, 60, 120, 180, 240, 300 degrees
    # for one orientation, or 30, 90, 150, 210, 270, 330 for the other.
    
    # Our lattice: x = sqrt(3)*(q + r/2), y = 1.5*r
    # The 6 lattice directions from center are at angles: 0, 60, 120, 180, 240, 300 degrees
    # (along the q, r, q+r axes). The normals to the sides of the container hexagon
    # should align with these directions for optimal packing.
    
    # For a regular hexagon with side L and center at origin, with one vertex pointing
    # along the x-axis (pointy-top in x), the side normals are at angles 30, 90, 150, 210, 270, 330.
    # Alternatively, with flat-top (side parallel to x-axis), normals are at 0, 60, 120, 180, 240, 300.
    
    # Let's use the flat-top orientation where normals are at 0, 60, 120, 180, 240, 300 degrees.
    # The inradius is L*sqrt(3)/2.
    # For a point (px, py), the distance to the side with normal at angle theta is
    # px*cos(theta) + py*sin(theta). The max over all 6 normals gives the required inradius.
    
    angles = [k * math.pi / 3.0 for k in range(6)]
    required_inradius = 0.0
    for x, y in centers:
        for theta in angles:
            d = x * math.cos(theta) + y * math.sin(theta)
            if d > required_inradius:
                required_inradius = d
    
    # Each unit hexagon extends 1 (circumradius) from its center in the direction of any normal
    # (since the circumradius is the max distance from center to any vertex).
    required_inradius += 1.0
    
    # Outer side length L: inradius = L * sqrt(3) / 2
    L = 2.0 * required_inradius / sqrt3
    
    # Build inner list: (x, y, angle_degrees)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon: center (0,0), side length L, rotation 0 degrees
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
