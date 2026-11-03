import math
import time
import itertools
import numpy as np

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors using convex polygon separation."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    # Quick bounding-circle rejection
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    verts1 = np.array(_hex_vertices(cx1, cy1, s1, a1))
    verts2 = np.array(_hex_vertices(cx2, cy2, s2, a2))
    # Check if any vertex of poly1 is strictly inside poly2
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    # Separating axis theorem: check all 12 edge normals
    for verts, other in [(verts1, verts2), (verts2, verts1)]:
        for i in range(6):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % 6]
            ex = bx - ax
            ey = by - ay
            length = math.sqrt(ex * ex + ey * ey)
            if length < 1e-15:
                continue
            nx = -ey / length
            ny = ex / length
            proj1 = verts[:, 0] * nx + verts[:, 1] * ny
            proj2 = other[:, 0] * nx + other[:, 1] * ny
            if proj1.max() <= proj2.min() + eps or proj2.max() <= proj1.min() + eps:
                return True
    return False

def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True

def _clip_polygon_by_line(poly, a, b, c):
    """
    Clip a convex polygon by the half-plane a*x + b*y + c <= 0.
    poly is a list of (x,y) tuples.
    Returns the clipped polygon as a list of (x,y) tuples.
    """
    if not poly:
        return []
    
    n = len(poly)
    out = []
    
    # Determine start point
    # We want to keep points where val <= 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        
        val1 = a * x1 + b * y1 + c
        val2 = a * x2 + b * y2 + c
        
        if val1 <= 0:
            out.append(poly[i])
        
        # Check for intersection
        if (val1 <= 0 and val2 > 0) or (val1 > 0 and val2 <= 0):
            # Intersect line segment (x1,y1)-(x2,y2) with a*x+b*y+c=0
            # t = -c / (a*(x2-x1) + b*(y2-y1))
            denom = a * (x2 - x1) + b * (y2 - y1)
            if abs(denom) > 1e-12:
                t = -c / denom
                x_int = x1 + t * (x2 - x1)
                y_int = y1 + t * (y2 - y1)
                out.append((x_int, y_int))
                
    return out

def _min_outer_side_fast(points, oangle_rad):
    """
    Find minimum side length of outer hexagon with rotation oangle_rad
    that contains all points.
    Uses binary search on side length S and feasibility check via polygon clipping.
    Returns (min_side, center_x, center_y)
    """
    if not points:
        return 0.0, 0.0, 0.0
    
    pts = np.array(points)
    
    # 6 normals for the outer hexagon
    # The outer hexagon sides are perpendicular to these normals.
    # The constraints are n_k . (p - c) <= A, where A is apothem = S * sqrt(3)/2
    # n_k . p - n_k . c <= A
    # n_k . p <= n_k . c + A
    # P_k <= n_k . c + A
    # -n_k . p <= -n_k . c + A
    # -Q_k <= -n_k . c + A  =>  Q_k >= n_k . c - A
    # So:
    # n_k . c >= P_k - A
    # n_k . c <= Q_k + A  (Wait, -n_k . c >= -Q_k - A => n_k . c <= Q_k + A)
    
    u_vectors = []
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        u_vectors.append((math.cos(theta), math.sin(theta)))
    
    P_vals = np.zeros(6)
    Q_vals = np.zeros(6)
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        P_vals[k] = np.max(projections)
        Q_vals[k] = np.min(projections)
    
    # We want to find min S such that there exists c=(cx,cy) satisfying:
    # n_k . c >= P_k - A(S)
    # -n_k . c >= Q_k + A(S)  (from -n_k . c >= -Q_k - A => n_k . c <= Q_k + A)
    # Wait, let's re-derive carefully.
    # Constraint: max_p (n_k . (p-c)) <= A
    # n_k . (p-c) <= A for all p
    # n_k . p - n_k . c <= A
    # max_p(n_k . p) - n_k . c <= A
    # P_k - n_k . c <= A  =>  n_k . c >= P_k - A
    
    # And for the opposite face (normal -n_k):
    # max_p (-n_k . (p-c)) <= A
    # -n_k . p + n_k . c <= A
    # -min_p(n_k . p) + n_k . c <= A
    # -Q_k + n_k . c <= A => n_k . c <= Q_k + A
    
    # So for each k:
    # P_k - A <= n_k . c <= Q_k + A
    
    # We binary search on S. A = S * sqrt(3) / 2.
    # For a given A, the feasible region for c is the intersection of 12 strips.
    # Strip k: P_k - A <= n_k . c <= Q_k + A
    
    def is_feasible(A):
        # Start with a large bounding box
        # Estimate a safe bound
        max_extent = np.max(np.abs(pts)) + 10
        poly = [(-max_extent, -max_extent), (max_extent, -max_extent), 
                (max_extent, max_extent), (-max_extent, max_extent)]
        
        for k in range(6):
            ux, uy = u_vectors[k]
            # Inequality 1: ux*cx + uy*cy >= P_k - A
            # => -ux*cx - uy*cy + (P_k - A) <= 0
            poly = _clip_polygon_by_line(poly, -ux, -uy, P_vals[k] - A)
            if not poly:
                return None
                
            # Inequality 2: ux*cx + uy*cy <= Q_k + A
            # => ux*cx + uy*cy - (Q_k + A) <= 0
            poly = _clip_polygon_by_line(poly, ux, uy, -(Q_vals[k] + A))
            if not poly:
                return None
                
        if not poly:
            return None
            
        # If feasible, return the centroid of the polygon as the center
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        return (cx, cy)

    # Lower bound: 0
    # Upper bound: large enough to contain everything
    max_dist = np.max(np.sqrt(pts[:,0]**2 + pts[:,1]**2))
    lo = 0.0
    hi = max_dist * 2.0 / (math.sqrt(3)/2) + 1.0
    
    # Binary search
    for _ in range(40):
        mid = (lo + hi) / 2.0
        A = mid * math.sqrt(3) / 2.0
        res = is_feasible(A)
        if res is not None:
            hi = mid
            best_center = res
        else:
            lo = mid
            
    # Final check
    A_final = hi * math.sqrt(3) / 2.0
    center = is_feasible(A_final)
    if center is None:
        # Fallback
        center = (0.0, 0.0)
        
    return hi, center[0], center[1]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = math.sqrt(3)

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # 11 positions: center + ring 1 (6) + 4 from ring 2
    pos_axial = [(0, 0)]
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_xy = None
    best_angle = 0.0
    best_outer_center = (0.0, 0.0)
    
    all_axial = pos_axial + ring2_candidates
    all_xy = [axial_to_xy(q, r) for q, r in all_axial]
    ring2_indices = list(range(7, len(all_xy)))
    
    # Phase 1: Coarse Search
    # Evaluate all combinations with coarse angle steps
    coarse_results = []
    
    for combo in itertools.combinations(ring2_indices, 4):
        if time.time() > deadline - 10: # Leave time for fine search
            break
            
        selected_indices = list(range(7)) + list(combo)
        full_xy = [all_xy[i] for i in selected_indices]
        
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in full_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        
        best_side_local = float('inf')
        best_theta_local = 0.0
        best_center_local = (0.0, 0.0)
        
        # Coarse angle grid
        for t_deg in range(0, 30, 2):
            if time.time() > deadline - 10:
                break
            t_rad = math.radians(t_deg)
            side, ox, oy = _min_outer_side_fast(inner_hexes_centered, t_rad)
            if side < best_side_local:
                best_side_local = side
                best_theta_local = t_deg
                best_center_local = (ox, oy)
                
        coarse_results.append((best_side_local, best_theta_local, best_center_local, full_xy))
        
    # Sort results
    coarse_results.sort(key=lambda x: x[0])
    
    # Phase 2: Fine Search on top candidates
    # Take top 20 or all if fewer
    top_candidates = coarse_results[:20]
    
    for side_est, theta_est, center_est, full_xy in top_candidates:
        if time.time() > deadline - 5:
            break
            
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in full_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        
        # Ternary search for fine angle
        # Range around theta_est +/- 2 degrees
        lo = max(0.0, theta_est - 2.0)
        hi = min(30.0, theta_est + 2.0)
        
        gr = (math.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = _min_outer_side_fast(inner_hexes_centered, math.radians(c))[0]
        fd = _min_outer_side_fast(inner_hexes_centered, math.radians(d))[0]
        
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            if fc < fd:
                hi = d
                d = c
                fd = fc
                c = hi - gr * (hi - lo)
                fc = _min_outer_side_fast(inner_hexes_centered, math.radians(c))[0]
            else:
                lo = c
                c = d
                fc = fd
                d = lo + gr * (hi - lo)
                fd = _min_outer_side_fast(inner_hexes_centered, math.radians(d))[0]
                
        best_theta_final = (lo + hi) / 2.0
        side_final, ox_final, oy_final = _min_outer_side_fast(inner_hexes_centered, math.radians(best_theta_final))
        
        if side_final < best_side:
            best_side = side_final
            best_angle = best_theta_final
            best_xy = full_xy
            best_outer_center = (ox_final + cx_mean, oy_final + cy_mean)

    # Fallback if nothing found
    if best_xy is None:
        if coarse_results:
             s, t, c, xy = coarse_results[0]
             best_side = s
             best_angle = t
             best_xy = xy
             cx_mean = sum(x for x, y in best_xy) / 11.0
             cy_mean = sum(y for x, y in best_xy) / 11.0
             best_outer_center = (c[0] + cx_mean, c[1] + cy_mean)
        else:
             # Very basic fallback
             best_xy = [all_xy[i] for i in range(7)]
             best_side = 4.0
             best_angle = 0.0
             best_outer_center = (0.0, 0.0)

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        scale = 1.01
        inner_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_xy]
        cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
        cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in [(h[0], h[1]) for h in inner_hexes]]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_fast(inner_hexes_centered, math.radians(best_angle))
        best_outer_center = (ox + cx_mean, oy + cy_mean)

    # Verify all inner hexagons are inside outer
    outer_hex = (best_outer_center[0], best_outer_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        best_side *= 1.005

    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    return inner, (best_outer_center[0], best_outer_center[1]), best_side, best_angle
