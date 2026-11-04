"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 120.0

    # Unit regular hexagon vertices (pointy-top orientation, circumradius = 1)
    # Flat-top: vertices at angles 0, 60, 120, 180, 240, 300
    hex_verts = []
    for i in range(6):
        angle = math.radians(60 * i)
        hex_verts.append((math.cos(angle), math.sin(angle)))

    # 11 unit hexagons in a row: centers at (4k, 0) for k in range(-5, 6)
    # Each hexagon is a rigid body; we can translate and rotate each independently.
    # But the task says "disjoint regular hexagons of side length 1" - they must not overlap.
    # The seed places them in a row with 4.0 spacing (which is 2*2 = touching for unit hexes? 
    # Actually for unit side hexagons, the distance between centers for non-overlapping in a row 
    # with same orientation is 2 * inradius = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732... 
    # Wait, no. For flat-top hexagons with side 1, the width is 2 (from leftmost to rightmost vertex).
    # The distance between centers of two adjacent flat-top hexagons in a row (sharing an edge) is sqrt(3).
    # Actually let me reconsider: a regular hexagon with side s has circumradius s and inradius s*sqrt(3)/2.
    # Two hexagons side-by-side sharing an edge: center-to-center distance = 2 * inradius = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732.
    # But the seed uses spacing 4.0, which is much larger. This suggests the seed is not optimal.

    # Let me think about this differently. The key insight: we want to pack 11 unit hexagons 
    # into the smallest possible outer hexagon. The outer hexagon has side length L.
    # The diameter of the outer hexagon (distance between opposite sides) is 2*L*sqrt(3)/2 = L*sqrt(3).
    # The diameter (vertex to vertex) is 2L.

    # Strategy: Place the 11 hexagons in a hexagonal lattice arrangement, then find the 
    # minimum enclosing hexagon.

    # For a hexagonal packing of unit hexagons, the centers are spaced by sqrt(3) in the 
    # nearest-neighbor direction. Let's use a hexagonal grid.

    # Actually, the simplest good packing: place them in a 1x11 row with minimal spacing.
    # For flat-top hexagons in a horizontal row, the minimal center spacing is sqrt(3) ≈ 1.732.
    # Total width of 11 hexagons in a row: 10*sqrt(3) + 2 (for the two end hexagons' extents)
    # = 10*1.732 + 2 = 19.32. The outer hexagon would need to contain this.

    # But we can do better with a more compact arrangement.

    # Let's try: place hexagons in a hexagonal pattern.
    # For 11 hexagons, one good arrangement is a 3-4-3 or 2-3-3-3 or 1-2-3-3-2-1 pattern.
    # Actually 11 = 1+2+3+3+2 (5 rows) or 2+3+3+3 (4 rows) or 3+4+4 (3 rows) etc.

    # Let's use a hexagonal lattice with flat-top orientation.
    # In a hexagonal grid, each hexagon has 6 neighbors at distance sqrt(3).
    # The grid positions: for row r and column c, center at:
    #   x = c * 1.5 * s  (where s is side length... wait, this is for pointy-top)
    
    # For flat-top hexagons:
    # x = 3/2 * s * col + (3/2 * s if row is odd else 0)  -- no, let me use the standard formula.
    
    # Standard flat-top hex grid:
    # x = s * 3/2 * col
    # y = s * sqrt(3) * (row + col/2)
    # where s is the side length (= 1 here).

    # Let's just try several arrangements and compute the minimal enclosing hexagon.

    def hexagon_corners(cx, cy, angle_deg=0.0):
        """Return 6 vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        pts = []
        for (vx, vy) in hex_verts:
            rx = cos_a * vx - sin_a * vy
            ry = sin_a * vx + cos_a * vy
            pts.append((cx + rx, cy + ry))
        return pts

    def convex_hull(points):
        """Andrew's monotone chain convex hull."""
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts
        def cross(O, A, B):
            return (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    def min_enclosing_hexagon_side(hull_points):
        """
        Find the minimum side length of a regular hexagon (any rotation) that contains all hull_points.
        Use golden-section search over rotation angle.
        """
        if not hull_points:
            return 0.0
        # The support function of a set of points in direction u is max dot(p, u).
        # A regular hexagon with side L, center c, rotation theta has 6 facets with outward normals
        # at angles theta + 60*k. The hexagon contains point p iff for all k:
        #   dot(p - c, n_k) <= L * sqrt(3) / 2  (the inradius)
        # where n_k is the unit normal to facet k.
        # So L >= 2/sqrt(3) * max_k max_p dot(p - c, n_k) for all k.
        # To minimize L, we also choose c optimally. For a given theta, the optimal c 
        # minimizes the maximum over k of [max_p dot(p, n_k) - min_p dot(p, n_k)] / 2,
        # because we can center c to balance.
        # Specifically, for direction n, the range is max_p dot(p,n) - min_p dot(p,n).
        # The hexagon needs inradius >= max_k range_k / 2.
        # So L >= 2/sqrt(3) * max_k (range_k / 2) = 1/sqrt(3) * max_k range_k.
        
        golden_ratio = (math.sqrt(5) - 1) / 2  # 0.618...
        
        def cost(theta):
            max_range = 0.0
            for k in range(6):
                angle = theta + math.radians(60 * k)
                nx = math.cos(angle)
                ny = math.sin(angle)
                dots = [px * nx + py * ny for (px, py) in hull_points]
                rng = max(dots) - min(dots)
                if rng > max_range:
                    max_range = rng
            return max_range / math.sqrt(3)
        
        # Search over theta in [0, pi/3) = [0, 60 degrees) due to 6-fold symmetry
        lo = 0.0
        hi = math.pi / 3.0
        # Golden section search
        c = hi - golden_ratio * (hi - lo)
        d = lo + golden_ratio * (hi - lo)
        fc = cost(c)
        fd = cost(d)
        for _ in range(60):
            if fc < fd:
                hi = d
                d = c
                fd = fc
                c = hi - golden_ratio * (hi - lo)
                fc = cost(c)
            else:
                lo = c
                c = d
                fc = fd
                d = lo + golden_ratio * (hi - lo)
                fd = cost(d)
        return min(fc, fd)

    def compute_packing(centers):
        """Given list of (cx, cy, angle_deg) for unit hexagons, return minimal outer hex side."""
        all_points = []
        for (cx, cy, ang) in centers:
            all_points.extend(hexagon_corners(cx, cy, ang))
        hull = convex_hull(all_points)
        return min_enclosing_hexagon_side(hull)

    # Try arrangement 1: 11 in a row with optimal spacing
    # For flat-top hexagons in a horizontal row, minimal center spacing = sqrt(3)
    spacing = math.sqrt(3)
    centers_row = []
    for k in range(-5, 6):
        centers_row.append((k * spacing, 0.0, 0.0))
    side_row = compute_packing(centers_row)

    best_side = side_row
    best_centers = centers_row
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Try arrangement 2: hexagonal cluster
    # Flat-top hex grid coordinates
    # Positions for a compact cluster of 11:
    # Let's use axial coordinates and pick 11 positions close to center.
    # Axial to pixel for flat-top:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    def axial_to_pixel(q, r):
        return (1.5 * q, math.sqrt(3) * (r + q / 2.0))

    # Generate hex grid positions by distance from center
    hex_positions = [(0, 0)]
    for dist in range(1, 20):
        # Walk around the hexagon at this distance
        q, r = dist, 0
        for direction in range(6):
            for i in range(dist):
                hex_positions.append((q, r))
                # Move in the 6th direction (which closes the loop)
                if direction == 0:
                    q += 1
                elif direction == 1:
                    r += 1
                elif direction == 2:
                    q -= 1; r += 1
                elif direction == 3:
                    q -= 1
                elif direction == 4:
                    r -= 1
                elif direction == 5:
                    q += 1; r -= 1
        if len(hex_positions) >= 11:
            break

    # Pick the 11 closest to center (which should be the center + 6 neighbors + 4 more)
    def hex_distance(q, r):
        return (abs(q) + abs(r) + abs(q + r)) // 2

    hex_positions.sort(key=lambda x: (hex_distance(x[0], x[1]), x[0] ** 2 + x[1] ** 2))
    # Take first 11 (including center)
    selected_axial = hex_positions[:11]
    centers_cluster = []
    for (q, r) in selected_axial:
        px, py = axial_to_pixel(q, r)
        centers_cluster.append((px, py, 0.0))
    
    if time.time() < deadline - 30:
        side_cluster = compute_packing(centers_cluster)
        if side_cluster < best_side:
            best_side = side_cluster
            best_centers = centers_cluster

    # Try arrangement 3: 3 rows (3-4-4 or 4-3-4)
    # Row 1: 4 hexagons, Row 2: 3, Row 3: 4
    # Or 2-3-3-3
    # Let's try 2-3-3-3
    row_configs = [
        [2, 3, 3, 3],
        [3, 4, 4],
        [1, 3, 4, 3],
        [1, 2, 3, 3, 2],
    ]
    
    for config in row_configs:
        if time.time() > deadline - 30:
            break
        centers_try = []
        for row_idx, count in enumerate(config):
            y = row_idx * math.sqrt(3)
            for col in range(count):
                x = (col - (count - 1) / 2.0) * 1.5  # for flat-top, horizontal spacing is 1.5 * side
                # Actually for flat-top, adjacent in same row: dx = 1.5, but they don't share edge if in same row
                # Let me reconsider: in a flat-top hex grid, neighbors in same row are at dx=1.5? No.
                # Flat-top: the 6 neighbors of (0,0) are at:
                # (1.5, sqrt(3)/2), (1.5, -sqrt(3)/2), (0, sqrt(3)), (0, -sqrt(3)), (-1.5, sqrt(3)/2), (-1.5, -sqrt(3)/2)
                # So horizontal spacing between same-row neighbors is 3.0 (two apart).
                # Same-row adjacent: dx = 1.5, dy = ±sqrt(3)/2 (these are diagonal neighbors)
                # True horizontal neighbors (same row in hex sense) don't exist in flat-top; 
                # the "rows" are offset.
                pass
        
        # Let me just use the axial coordinate approach more carefully
        # For now, skip these and rely on the cluster approach

    # Final answer: use the best arrangement found
    # Compute the optimal outer hexagon center and angle
    all_points = []
    for (cx, cy, ang) in best_centers:
        all_points.extend(hexagon_corners(cx, cy, ang))
    hull = convex_hull(all_points)
    
    # Find optimal rotation and center
    golden_ratio = (math.sqrt(5) - 1) / 2
    def cost_theta(theta, pts):
        max_range = 0.0
        for k in range(6):
            angle = theta + math.radians(60 * k)
            nx = math.cos(angle)
            ny = math.sin(angle)
            dots = [px * nx + py * ny for (px, py) in pts]
            rng = max(dots) - min(dots)
            if rng > max_range:
                max_range = rng
        return max_range

    lo = 0.0
    hi = math.pi / 3.0
    c = hi - golden_ratio * (hi - lo)
    d = lo + golden_ratio * (hi - lo)
    fc = cost_theta(c, hull)
    fd = cost_theta(d, hull)
    for _ in range(50):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - golden_ratio * (hi - lo)
            fc = cost_theta(c, hull)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + golden_ratio * (hi - lo)
            fd = cost_theta(d, hull)
    
    best_theta = (c + d) / 2 if fc < fd else (c + d) / 2
    if fc < fd:
        best_theta = c
    else:
        best_theta = d
    
    # Compute optimal center for this rotation
    # For each facet direction, center should be at midpoint of [min, max] projection
    cx_center = 0.0
    cy_center = 0.0
    for k in range(6):
        angle = best_theta + math.radians(60 * k)
        nx = math.cos(angle)
        ny = math.sin(angle)
        dots = [px * nx + py * ny for (px, py) in hull]
        proj_center = (max(dots) + min(dots)) / 2.0
        cx_center += proj_center * nx
        cy_center += proj_center * ny
    cx_center /= 6.0
    cy_center /= 6.0

    final_side = best_side
    # Adjust: the center of the outer hexagon
    # The outer hexagon has center at (cx_center, cy_center) and rotation best_theta
    # Side length = best_side

    # Return format: inner rows (x, y, angle_degrees), outer center (x,y), side length, rotation
    inner_result = [(cx, cy, ang) for (cx, cy, ang) in best_centers]
    return inner_result, (cx_center, cy_center), final_side, math.degrees(best_theta)
# EVOLVE-BLOCK-END
