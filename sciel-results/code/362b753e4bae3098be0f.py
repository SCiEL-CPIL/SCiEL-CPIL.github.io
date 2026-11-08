"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Place unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Strategy: Use a hexagonal (honeycomb) lattice packing. Unit hexagons tile the plane
    perfectly. We place hexagons on a honeycomb lattice and find the minimal enclosing
    regular hexagon.
    
    For a hexagonal lattice of unit hexagons:
    - The centers of adjacent hexagons are at distance sqrt(3) apart (since the inradius of a unit hexagon is sqrt(3)/2, and adjacent hexagons share an edge, so center-to-center distance = 2 * inradius = sqrt(3)).
    
    Wait, let me reconsider. A regular hexagon with side length 1 has:
    - Inradius (apothem) = sqrt(3)/2
    - Circumradius = 1
    - Adjacent hexagon centers in a tiling are at distance 2 * inradius = sqrt(3)
    
    The honeycomb lattice vectors for hexagon centers:
    - v1 = (sqrt(3), 0)
    - v2 = (sqrt(3)/2, 3/2)
    
    So a hexagon at lattice position (i, j) has center:
    x = i * sqrt(3) + j * sqrt(3)/2
    y = j * 3/2
    
    For the minimal enclosing hexagon, we need to find the smallest regular hexagon (with some rotation) that contains all the inner hexagons.
    
    Actually, let's think about this differently. If we place hexagons in a hexagonal arrangement (like a larger hexagon made of smaller hexagons), the optimal packing for N hexagons in a hexagonal shape uses a "hexagonal number" arrangement.
    
    For a hexagonal arrangement with "radius" r (number of hexagons from center to edge), the total number of hexagons is 1 + 3*r*(r+1) = 3r^2 + 3r + 1.
    
    For N=11, let's check: 
    r=1: 1+6=7
    r=2: 1+18=19
    So 11 doesn't fit a perfect hexagonal number.
    
    Let me try a different approach. Place hexagons on a honeycomb lattice and find the minimal enclosing hexagon.
    
    For a regular hexagon of side length S, the distance from center to each vertex is S, and the distance from center to each edge midpoint is S*sqrt(3)/2.
    
    To contain a unit hexagon centered at position (cx, cy) with rotation angle a, the outer hexagon must contain all 6 vertices of the inner hexagon.
    
    The vertices of a unit hexagon centered at (cx, cy) with rotation a are at:
    (cx + cos(a + k*pi/3), cy + sin(a + k*pi/3)) for k=0..5
    
    For the outer hexagon centered at (0,0) with side length S and rotation b, a point (x,y) is inside if:
    For each of the 6 edges, the point must be on the correct side.
    
    This is complex. Let me simplify by assuming the outer hexagon is axis-aligned (rotation 0) and the inner hexagons are also axis-aligned.
    
    For an axis-aligned regular hexagon of side length S centered at origin:
    - The vertices are at (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    - The half-width in x is S
    - The half-height in y is S*sqrt(3)/2
    - But it's not just a rectangle; the hexagon has slanted sides.
    
    A point (x,y) is inside an axis-aligned hexagon of side S if:
    |x| <= S
    |y| <= S*sqrt(3)/2
    AND |x| + |y|/sqrt(3) <= S  (this is the constraint from the slanted edges)
    
    Actually, let me be more careful. The six inequalities for an axis-aligned hexagon centered at origin with side S:
    1. x <= S
    2. x >= -S  
    3. sqrt(3)*y <= S*sqrt(3)  =>  y <= S  ... no wait.
    
    Let me use the standard form. A regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees from center, with circumradius R (= S for side length S):
    Vertices: (R*cos(k*pi/3), R*sin(k*pi/3)) for k=0..5
    
    The edges connect consecutive vertices. The half-space representation:
    For each edge, the normal points outward.
    
    Edge from vertex 0 to vertex 1: from (R,0) to (R/2, R*sqrt(3)/2)
    The outward normal direction is (cos(30°), sin(30°)) = (sqrt(3)/2, 1/2)
    The distance from center to this edge is R*cos(30°) = R*sqrt(3)/2
    So the inequality is: (sqrt(3)/2)*x + (1/2)*y <= R*sqrt(3)/2
    Simplify: sqrt(3)*x + y <= R*sqrt(3)
    
    Similarly for all 6 edges. The 6 inequalities are:
    1. x <= R
    2. sqrt(3)*x/2 + y/2 <= R*sqrt(3)/2  =>  sqrt(3)*x + y <= R*sqrt(3)
    3. -sqrt(3)*x + y <= R*sqrt(3)
    4. -x <= R  =>  x >= -R
    5. -sqrt(3)*x - y <= R*sqrt(3)
    6. sqrt(3)*x - y <= R*sqrt(3)
    
    Wait, let me redo this more carefully. For a regular hexagon with circumradius R (side length = R), the apothem (inradius) is R*sqrt(3)/2.
    
    The 6 supporting lines (edges) are at distance R*sqrt(3)/2 from center, with normals at angles 0°, 60°, 120°, 180°, 240°, 300°.
    
    So the inequalities are:
    For k = 0..5:
      cos(k*pi/3) * x + sin(k*pi/3) * y <= R * sqrt(3)/2
    
    This is cleaner. So a point (x,y) is inside iff for all k=0..5:
      cos(k*pi/3) * x + sin(k*pi/3) * y <= R * sqrt(3)/2
    
    Now, for a unit hexagon (side 1, circumradius 1) centered at (cx, cy) with rotation a, its vertices are at:
      (cx + cos(a + k*pi/3), cy + sin(a + k*pi/3)) for k=0..5
    
    For the outer hexagon to contain this inner hexagon, all 6 vertices of the inner hexagon must satisfy the 6 inequalities of the outer hexagon.
    
    This is a lot of constraints. But if we assume all inner hexagons have the same rotation as the outer hexagon (a = b), then the problem simplifies.
    
    Let's assume the outer hexagon has rotation 0, and all inner hexagons also have rotation 0. Then the inner hexagon vertices are at:
      (cx + cos(k*pi/3), cy + sin(k*pi/3)) for k=0..5
    
    For the outer hexagon centered at origin with circumradius R:
    For each inner vertex (vx, vy), we need:
      cos(k*pi/3) * vx + sin(k*pi/3) * vy <= R * sqrt(3)/2  for all k=0..5
    
    The maximum of cos(k*pi/3)*vx + sin(k*pi/3)*vy over k is the support function of the point in the hexagonal norm, which is max over k of the dot product.
    
    Actually, the condition that a point (vx,vy) is inside the outer hexagon is:
      max_{k=0..5} [cos(k*pi/3)*vx + sin(k*pi/3)*vy] <= R * sqrt(3)/2
    
    So R >= (2/sqrt(3)) * max_{k=0..5} [cos(k*pi/3)*vx + sin(k*pi/3)*vy]
    
    For all inner hexagons and all their vertices, we need R to be at least the maximum of these values.
    
    So the minimal R is:
      R = (2/sqrt(3)) * max over all inner hexagons h, all vertices v of h, all k=0..5 of [cos(k*pi/3)*v.x + sin(k*pi/3)*v.y]
    
    Since the inner hexagons are axis-aligned (rotation 0), their vertices are at (cx + cos(j*pi/3), cy + sin(j*pi/3)) for j=0..5.
    
    So for a given center (cx, cy), the contribution to R is:
      (2/sqrt(3)) * max_{j=0..5, k=0..5} [cos(k*pi/3)*(cx + cos(j*pi/3)) + sin(k*pi/3)*(cy + sin(j*pi/3))]
    
    This can be computed for each candidate center.
    
    Now, for the honeycomb lattice, the centers are at:
      (i*sqrt(3) + j*sqrt(3)/2, j*3/2) for integers i, j
    
    We want to choose a set of N lattice points (centers) such that the resulting R is minimized.
    
    For N=11, let's try a hexagonal arrangement. The centers of a hexagonal cluster of radius 1 (7 hexagons) are at:
      (0,0), (±sqrt(3), 0), (±sqrt(3)/2, ±3/2), (0, ±3)
      Wait, let me list them properly.
    
    Lattice positions (i,j) for a hexagonal cluster:
    Center: (0,0)
    Ring 1 (6 hexagons): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    
    Centers:
    (0,0) -> (0, 0)
    (1,0) -> (sqrt(3), 0)
    (0,1) -> (sqrt(3)/2, 3/2)
    (-1,1) -> (-sqrt(3)/2, 3/2)
    (-1,0) -> (-sqrt(3), 0)
    (0,-1) -> (-sqrt(3)/2, -3/2)
    (1,-1) -> (sqrt(3)/2, -3/2)
    
    That's 7 hexagons. For 11, we need 4 more. Let's add some from ring 2.
    
    Ring 2 positions: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)
    Wait, that's not right either. Let me use the standard hexagonal coordinate system.
    
    Actually, let me just try to place 11 hexagons in a compact arrangement and compute R.
    
    Let me try: center + 6 in ring 1 + 4 more in ring 2.
    
    Ring 2 positions (continuing the hexagonal pattern):
    (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)
    
    Hmm, this is getting complicated. Let me just write code to try various arrangements and compute R.
    
    Actually, let me just try a simple approach: place hexagons in a hexagonal grid pattern and compute the required R.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    sqrt3 = math.sqrt(3)
    
    # Honeycomb lattice vectors
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2)
    
    def lattice_center(i, j):
        x = i * sqrt3 + j * sqrt3 / 2
        y = j * 3 / 2
        return (x, y)
    
    def compute_R_for_centers(centers):
        """Compute the minimal circumradius R of an axis-aligned outer hexagon that contains all unit hexagons at given centers."""
        max_val = 0.0
        for (cx, cy) in centers:
            for j in range(6):
                vx = cx + math.cos(j * math.pi / 3)
                vy = cy + math.sin(j * math.pi / 3)
                for k in range(6):
                    val = math.cos(k * math.pi / 3) * vx + math.sin(k * math.pi / 3) * vy
                    if val > max_val:
                        max_val = val
        return (2.0 / sqrt3) * max_val
    
    # Try hexagonal cluster arrangements
    # For N=11, try center + ring 1 (6) + 4 from ring 2
    
    # Generate candidate sets of lattice points
    best_R = float('inf')
    best_centers = None
    
    # Hexagonal cluster: include all points with hexagonal distance <= r from origin
    def hex_distance(i, j):
        # Convert axial to cube coordinates
        x = i
        z = j
        y = -x - z
        return (abs(x) + abs(y) + abs(z)) // 2
    
    # Try clusters of various shapes
    # First, try perfect hexagonal clusters and add extra points
    
    for max_dist in range(0, 5):
        if time.time() > deadline:
            break
        points = []
        for i in range(-max_dist, max_dist + 1):
            for j in range(-max_dist, max_dist + 1):
                if hex_distance(i, j) <= max_dist:
                    points.append((i, j))
        
        if len(points) < 11:
            continue
        if len(points) > 11:
            # Try to select 11 from these
            # For now, just use the first 11 (not optimal but a start)
            points = points[:11]
        
        centers = [lattice_center(i, j) for (i, j) in points]
        R = compute_R_for_centers(centers)
        if R < best_R:
            best_R = R
            best_centers = centers[:]
    
    # Also try some manual arrangements
    # Center + 6 neighbors + 4 more
    base_points = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Add 4 more from ring 2
    ring2_candidates = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1), (2, -2)]
    
    # Try different combinations of 4 from ring 2
    import itertools
    for combo in itertools.combinations(range(len(ring2_candidates)), 4):
        if time.time() > deadline:
            break
        points = base_points + [ring2_candidates[i] for i in combo]
        centers = [lattice_center(i, j) for (i, j) in points]
        R = compute_R_for_centers(centers)
        if R < best_R:
            best_R = R
            best_centers = centers[:]
    
    if best_centers is None:
        # Fallback: just place them in a line
        best_centers = [lattice_center(i, 0) for i in range(-5, 6)]
        best_R = compute_R_for_centers(best_centers)
    
    # Construct the inner hexagons
    inner = []
    for (cx, cy) in best_centers:
        inner.append([cx, cy, 0.0])
    
    # The outer hexagon: center at (0,0), side length = best_R (circumradius = side length for regular hexagon), rotation 0
    # Wait, for a regular hexagon, the side length equals the circumradius. So side = R.
    side = best_R
    
    return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-END
