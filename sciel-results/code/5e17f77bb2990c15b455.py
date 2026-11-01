"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Strategy: Use hexagonal lattice positions for unit hexagons. The centers of
    unit hexagons in a hexagonal tiling form a triangular lattice. We search over
    compact 11-cell polyhexes and compute the minimum enclosing regular hexagon.
    
    For hexagons with flat-top orientation, the lattice vectors are:
    v1 = (3/2, √3/2)  -> distance √3 between adjacent hexagon centers
    v2 = (3/2, -√3/2)
    
    Actually, for unit hexagons (side=1), the center-to-center distance in a
    hexagonal tiling is 2 * (√3/2) = √3 ≈ 1.732.
    
    Let me use axial coordinates for hexagonal grid. A unit hexagon at axial (q,r)
    has center at:
    x = √3 * (q + r/2)
    y = 3/2 * r
    
    Wait, let me be more careful. For a flat-top hexagon with side length 1:
    - The "radius" (center to vertex) is 1
    - The "inradius" (center to edge midpoint) is √3/2
    - Adjacent hexagon centers are 2 * inradius = √3 apart
    
    Using axial coordinates (q, r) with pointy-top:
    x = √3 * (q + r/2)
    y = 3/2 * r
    
    For flat-top:
    x = 3/2 * q
    y = √3 * (r + q/2)
    
    Let me use a simpler approach: generate candidate positions on a hex grid,
    find all valid 11-subsets that form connected clusters, compute the minimum
    enclosing hexagon side length for each, and return the best.
    
    However, brute force over all 11-subsets is infeasible. Instead, I'll use
    known optimal packings or a smarter search.
    
    Known result: For packing n unit hexagons in a hexagon, the optimal side
    length for small n follows specific patterns. For n=11, the optimal
    arrangement is a 3x3 hexagonal block minus 8 corners, or similar.
    
    Let me try a direct approach: generate positions in a hexagonal grid of
    sufficient size, then use a heuristic to find a compact 11-hexagon cluster,
    and compute the enclosing hexagon.
    """
    
    # Generate hex grid positions using axial coordinates
    # For flat-top hexagons with side 1:
    # center(q, r) = (1.5*q, sqrt(3)*(r + 0.5*q))
    
    sqrt3 = np.sqrt(3)
    
    def hex_center(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return (x, y)
    
    # Generate a range of axial coordinates
    # We need enough to fit 11 hexagons compactly
    max_coord = 5
    positions = []
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            positions.append((q, r))
    
    # Convert to Cartesian
    pos_cart = [hex_center(q, r) for q, r in positions]
    pos_axial = positions
    
    # We want to find 11 positions that form a connected cluster minimizing
    # the enclosing hexagon side length.
    
    # For a set of points, the minimum enclosing regular hexagon (with any rotation)
    # side length can be computed. But this is complex.
    
    # Alternative: For hexagonal packings, the optimal enclosing hexagon is
    # aligned with the lattice. The side length is determined by the extent
    # of the cluster in the three lattice directions.
    
    # Let me use a different approach: try known good configurations.
    
    # Configuration 1: Hexagonal number pattern
    # A hexagon of "radius" k (k hexagons from center to edge) has 3k(k-1)+1 hexagons
    # k=1: 1, k=2: 7, k=3: 19
    # For 11, we take a k=3 hexagon and remove 8 hexagons from the corners/edges
    
    # Let me enumerate all 11-subsets of a k=3 hexagonal cluster (19 positions)
    # and find the one with minimum enclosing hexagon.
    
    # k=3 hexagonal cluster in axial coords: all (q,r) with |q|<=2, |r|<=2, |q+r|<=2
    # Wait, that's not right. Let me use the standard hex distance.
    
    # Hex distance from (0,0) to (q,r) = max(|q|, |r|, |q+r|)
    # Actually for axial: hex_dist(q,r) = (|q| + |r| + |q+r|) / 2
    
    k3_positions = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            dist = (abs(q) + abs(r) + abs(q + r)) // 2
            if dist <= 2:  # radius 2 gives 19 hexagons
                k3_positions.append((q, r))
    
    # That should give 19 positions. Let me verify count.
    # Actually let me just generate and count.
    
    # For efficiency, let me try a smaller search space.
    # Use radius 2 hex cluster (19 positions) and find best 11-subset.
    
    # But C(19,11) = 75582, which is manageable.
    
    def enclosing_hex_side(carts):
        """Compute minimum enclosing regular hexagon side length for a set of 2D points.
        For a hexagon with flat-top orientation centered at origin, a point (x,y) is
        inside if max(|x|, |y|/sqrt(3), |x/2 - y*sqrt(3)/2|) <= inradius * 2 / sqrt(3)... 
        
        Actually, let me think differently. For a regular hexagon with side length S,
        the inradius is S*sqrt(3)/2. The hexagon can be described as the intersection
        of 6 half-planes. For a flat-top hexagon centered at (cx, cy):
        
        The six normals are at angles 0, 60, 120, 180, 240, 300 degrees.
        A point is inside if for each normal n_i, dot((x,y) - (cx,cy), n_i) <= inradius
        
        where inradius = S * sqrt(3) / 2.
        
        So S = 2 * max_i |dot((x,y) - (cx,cy), n_i)| / sqrt(3)
        
        To minimize S over center (cx,cy) and rotation, we need to solve a convex
        optimization. For simplicity, let's assume the hexagon is aligned with the
        lattice (flat-top) and find the best center.
        """
        # For a set of points, find the minimum S such that there exists a center
        # (cx, cy) where all points are inside a flat-top hexagon of side S.
        
        # The constraint is: for each point p_i and each of the 6 directions,
        # |dot(p_i - c, n_j)| <= S * sqrt(3) / 2
        
        # This is a linear programming problem in (cx, cy, S).
        
        # For simplicity, let's compute the bounding box in the 6 hex directions
        # and take the max extent.
        
        # The 6 unit normals for flat-top hexagon (pointing to edge midpoints):
        # Angles: 0, 60, 120, 180, 240, 300
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        
        # Project points onto each normal
        # For each normal direction, the extent is (max_proj - min_proj)
        # The inradius must be at least half the extent in each direction
        # So S * sqrt(3)/2 >= max_i (extent_i / 2)
        # S >= 2 * max_i(extent_i) / sqrt(3)
        
        pts = np.array(carts)
        extents = []
        for n in normals:
            projs = pts @ n
            extent = projs.max() - projs.min()
            extents.append(extent)
        
        max_extent = max(extents)
        S = 2 * max_extent / sqrt3
        
        return S
    
    # Now search over all 11-subsets of the 19 positions
    best_S = float('inf')
    best_subset = None
    
    from itertools import combinations
    
    for subset in combinations(range(len(k3_positions)), 11):
        carts = [hex_center(*k3_positions[i]) for i in subset]
        S = enclosing_hex_side(carts)
        if S < best_S:
            best_S = S
            best_subset = subset
    
    # Get the best configuration
    best_carts = [hex_center(*k3_positions[i]) for i in best_subset]
    
    # Now we need to output: inner rows (x, y, angle_degrees), outer center, side, angle
    # The inner hexagons are unit hexagons. Their angle is 0 (flat-top) by default.
    # We can rotate them, but for simplicity keep them aligned.
    
    inner = []
    for (x, y) in best_carts:
        inner.append([x, y, 0.0])
    
    # Outer hexagon center: use the centroid of the inner hexagons
    cx = np.mean([p[0] for p in best_carts])
    cy = np.mean([p[1] for p in best_carts])
    
    return inner, (float(cx), float(cy)), float(best_S), 0.0
# EVOLVE-BLOCK-END

