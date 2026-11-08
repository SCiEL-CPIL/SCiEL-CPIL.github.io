"""Packing unit regular hexagons inside a minimal enclosing regular hexagon.

Approach:
- Place unit regular hexagons on a honeycomb lattice (hexagonal tiling).
- The lattice vectors for a regular hexagonal tiling of unit hexagons:
  For a regular hexagon with side length 1, the distance between centers of
  adjacent hexagons in the tiling is sqrt(3) (the inradius * 2 = 2 * (sqrt(3)/2) = sqrt(3)).
  
  Actually, let's be precise. A regular hexagon with side length s has:
  - Inradius (apothem) = s * sqrt(3)/2
  - Circumradius = s
  - Distance between centers of adjacent hexagons in a perfect tiling = 2 * inradius = s * sqrt(3)
  
  For unit hexagons (s=1), center-to-center distance = sqrt(3).
  
  The hexagonal lattice can be described by two basis vectors:
  e1 = (sqrt(3), 0)
  e2 = (sqrt(3)/2, 3/2)
  
  Wait, let me reconsider. In a standard hexagonal tiling where hexagons have flat tops:
  - Horizontal spacing between centers: sqrt(3)
  - Vertical spacing: 1.5 (for alternating rows)
  
  Actually, for regular hexagons with side length 1 oriented with a flat top:
  - Width (point to point horizontally) = 2
  - Height (flat to flat vertically) = sqrt(3)
  - Distance between centers horizontally (same row) = sqrt(3)
  - Distance between centers in adjacent rows = 1.5 (vertical offset) and sqrt(3)/2 (horizontal offset)
  
  Let me use the axial coordinate system for hexagonal grids.
  
  For a hexagonal tiling of unit hexagons, the center positions can be given by:
  For integer coordinates (q, r) in axial coordinates:
  x = sqrt(3) * (q + r/2)
  y = 1.5 * r
  
  This gives a perfect tiling with no overlaps.
  
- We need to find the minimal enclosing regular hexagon. Since both the inner hexagons
  and the outer container are regular hexagons, we can exploit symmetry.
  
- The key insight: if we place N unit hexagons on the honeycomb lattice centered at the origin,
  the enclosing regular hexagon (same orientation) needs to have a side length large enough
  to contain all of them.
  
- For a regular hexagon with side length S centered at origin with flat-top orientation:
  The vertices are at distance S from center, at angles 0, 60, 120, 180, 240, 300 degrees.
  A point (x, y) is inside if for all 6 directions, the projection is within bounds.
  
  For a flat-top regular hexagon with side length S centered at origin:
  The six sides are at distance S*sqrt(3)/2 from center (the inradius).
  
  A point (x,y) is inside the hexagon if:
  |x| <= S * sqrt(3)/2  (for left/right sides... wait, no)
  
  Let me think more carefully. A regular hexagon with flat top, side length S, centered at origin:
  Vertices at: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
  
  The six half-planes defining the interior:
  1. x <= S
  2. x/2 + sqrt(3)*y/2 <= S/2  =>  x + sqrt(3)*y <= S
  3. -x/2 + sqrt(3)*y/2 <= S/2  =>  -x + sqrt(3)*y <= S
  4. -x <= S  =>  x >= -S
  5. -x/2 - sqrt(3)*y/2 <= S/2  =>  -x - sqrt(3)*y <= S
  6. x/2 - sqrt(3)*y/2 <= S/2  =>  x - sqrt(3)*y <= S
  
  So a point (x,y) is inside if all of these are satisfied.
  
  For each inner hexagon center (cx, cy), the inner hexagon itself extends to distance 1 from its center (circumradius = 1). So the outer hexagon must contain all points at distance up to 1 from each inner center.
  
  The constraint is: for each inner hexagon center (cx, cy), and for each of the 6 half-plane constraints, we need:
  max over the inner hexagon of the linear form <= S
  
  Since the inner hexagon is convex and the linear form is linear, the max occurs at a vertex. But more simply, the maximum of a linear form a*x + b*y over a disk of radius 1 centered at (cx,cy) is a*cx + b*cy + sqrt(a^2+b^2).
  
  For our constraints, the linear forms have coefficients of magnitude 1 or sqrt(3)/2, etc. Let me just compute the required S for each constraint.
  
  For constraint 1: x <= S. Max x over inner hex at (cx,cy) is cx + 1 (since circumradius is 1, and the max x-direction extent is 1). So S >= cx + 1.
  For constraint 4: -x <= S => x >= -S. Min x is cx - 1. So S >= 1 - cx.
  For constraint 2: x + sqrt(3)*y <= S. Max of x+sqrt(3)*y over disk radius 1 at (cx,cy) is cx + sqrt(3)*cy + 2. So S >= cx + sqrt(3)*cy + 2.
  Similarly for others.
  
  So S must be at least the max over all inner hexagons and all 6 constraints.
  
- Strategy: Generate a set of lattice points (hexagon centers) forming a roughly hexagonal cluster of N hexagons, compute the required S, and return.

- To minimize S, we want the cluster to be as compact as possible, i.e., hexagonal shape.

"""

import numpy as np
import math

N = 11

def construct():
    """
    Place N unit regular hexagons on a honeycomb lattice to minimize the
    enclosing regular hexagon side length.
    
    Returns:
        inner: list of [x, y, angle_degrees] for each inner hexagon
        center: (x, y) of outer hexagon center
        side: side length of outer hexagon
        angle: rotation of outer hexagon (degrees)
    """
    # Generate hexagonal cluster of N hexagons on honeycomb lattice
    # Using axial coordinates (q, r), center position:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # For a hexagonal cluster of "radius" k (in hex grid terms), the number of cells is 3k(k+1)+1
    # We want to pick the largest hexagonal cluster with count <= N, then add remaining cells.
    
    sqrt3 = math.sqrt(3)
    
    # Generate candidate positions in hexagonal cluster order
    # Start from center (0,0) and expand outward
    def hex_distance(q, r):
        return (abs(q) + abs(r) + abs(q + r)) // 2
    
    # Generate all lattice points sorted by distance from origin
    max_k = 10
    points = []
    for r in range(-max_k, max_k + 1):
        for q in range(-max_k, max_k + 1):
            d = hex_distance(q, r)
            points.append((d, q, r))
    
    points.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Take the first N points
    selected = points[:N]
    
    # Convert to Cartesian coordinates
    centers = []
    for _, q, r in selected:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Compute the minimum enclosing regular hexagon side length
    # For a flat-top regular hexagon centered at (0,0) with side length S:
    # A point (x,y) with an inner hexagon of circumradius 1 at that point
    # requires S >= max over 6 constraints
    
    def required_side(cx, cy):
        # Six constraints, each giving a lower bound on S
        # Constraint 1: x <= S  => S >= cx + 1
        # Constraint 4: x >= -S => S >= 1 - cx
        # Constraint 2: x + sqrt(3)*y <= S => S >= cx + sqrt(3)*cy + 2
        # Constraint 5: x + sqrt(3)*y >= -S => S >= -cx - sqrt(3)*cy + 2
        # Constraint 3: -x + sqrt(3)*y <= S => S >= -cx + sqrt(3)*cy + 2
        # Constraint 6: -x + sqrt(3)*y >= -S => S >= cx - sqrt(3)*cy + 2
        
        bounds = [
            cx + 1,
            1 - cx,
            cx + sqrt3 * cy + 2,
            -cx - sqrt3 * cy + 2,
            -cx + sqrt3 * cy + 2,
            cx - sqrt3 * cy + 2
        ]
        return max(bounds)
    
    S = max(required_side(cx, cy) for cx, cy in centers)
    
    # Build inner list
    inner = [[x, y, 0.0] for x, y in centers]
    
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
