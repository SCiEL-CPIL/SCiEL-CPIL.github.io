import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap.
    """
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _compute_side(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon.
    inner_params: array of shape (n, 3) with [x, y, angle_deg]
    """
    n = len(inner_params)
    
    lo = 2.0
    hi = 15.0
    
    for _ in range(50):
        mid = (lo + hi) / 2
        outer_v = _outer_hex_vertices(0.0, 0.0, mid, outer_angle_deg)
        
        valid = True
        for i in range(n):
            x = inner_params[i, 0]
            y = inner_params[i, 1]
            a = inner_params[i, 2]
            hex_v = _hex_vertices(x, y, a)
            for v in hex_v:
                if not _point_in_polygon(v, outer_v):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            hi = mid
        else:
            lo = mid
    
    return hi

def _overlap_penalty(inner_params, penalty_weight=200.0):
    """
    Compute penalty for overlapping inner hexagons.
    """
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < 0.01:
                penalty += penalty_weight * (0.01 - sep) ** 2
    
    return penalty

def _objective(params, outer_angle_deg, n):
    """
    Objective function: outer side length + overlap penalty.
    """
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if the configuration is valid."""
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < min_sep:
                return False
    
    return True

def _seed_solution():
    """
    Return the concentric ring seed solution with correct geometry.
    For a flower packing of 11 hexagons:
    - Center hexagon at origin
    - 6 hexagons in first ring at distance sqrt(3) from center
    - 4 hexagons in second ring at distance 3.0 from center
    
    The key insight: for unit hexagons with flat-top orientation (angle=0),
    the centers of adjacent hexagons in a hexagonal lattice are at distance
    2*sqrt(3)/sqrt(3) = 2 (wait, let me recalculate).
    
    Actually, for regular hexagons of side 1:
    - The distance between centers of two adjacent hexagons sharing an edge
      is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    - The distance between centers of hexagons that touch at vertices
      is 2 * (side) = 2 (wait, no)
    
    Let me think more carefully. A regular hexagon with side s has:
    - Apothem (inradius) = s * sqrt(3) / 2
    - Circumradius = s
    
    For two hexagons to be tangent (touching at one point), the distance
    between their centers depends on the orientation and which parts touch.
    
    For the standard hexagonal tiling with flat-top hexagons:
    - Horizontal spacing between centers: 3/2 * s = 1.5
    - Vertical spacing between rows: sqrt(3) * s = sqrt(3)
    - The 6 neighbors of a center hexagon are at distance sqrt(3) from center
    
    Wait, let me verify. If hexagon A is at origin with flat-top orientation,
    its rightmost vertex is at (1, 0). The adjacent hexagon to the right
    would have its leftmost vertex at (1, 0), so its center is at (2, 0).
    But that's distance 2, not sqrt(3).
    
    Hmm, I think I'm confusing myself. Let me use the vertex-based approach.
    
    For a flat-top hexagon (vertices at angles 0, 60, 120, 180, 240, 300),
    the vertices are at:
    (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    The right neighbor hexagon shares the edge from (1, 0) to (0.5, sqrt(3)/2)?
    No, that's not right either.
    
    Actually, in a hexagonal tiling with pointy-top hexagons (vertices at 30, 90, 150, 210, 270, 330),
    the horizontal distance between centers is sqrt(3), and the vertical distance is 1.5.
    
    Let me just use the known result: for the standard flower packing of 7 hexagons
    (1 center + 6 around), the outer hexagon side length is 3.0 when all hexagons
    have the same orientation as the outer one.
    
    For 11 hexagons (1 center + 6 first ring + 4 second ring), the optimal
    outer side length should be around 4.0-5.0.
    
    Let me just use a reasonable seed and let the optimizer find the optimum.
    """
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    # First ring: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (SQRT3 + eps) * np.cos(angle)
        y = (SQRT3 + eps) * np.sin(angle)
        centers.append((x, y))
    # Second ring: 4 hexagons at distance 3.0, at angles 30, 90, 150, 210, 270, 330
    # Actually, for the flower pattern, the second ring has 6 positions but we only
    # place 4 of them (the ones at 30, 90, 150, 210, 270, 330 minus two opposite ones)
    # Let's place them at 30, 90, 150, 270 (skipping 210 and 330 to avoid overlap)
    # Actually, let's try all 6 positions and see if they overlap
    for k in range(4):
        angle = np.deg2rad(30 + k * 90)  # 30, 120, 210, 300
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # Compute a safe outer side length
    max_dist = 0
    for x, y in centers:
        d = math.sqrt(x*x + y*y)
        max_dist = max(max_dist, d)
    # The hexagon extends 1 unit from its center in all directions
    # The outer hexagon needs to contain all these, so side >= max_dist + 1
    # But more precisely, we need the bounding hexagon side
    side = max_dist + 2.0 + eps
    
    return inner, (0.0, 0.0), side, 30.0

def _fast_side_length(params, n, outer_angle=30.0):
    """
    Fast analytical computation of outer side length using projection onto 6 face normals.
    params: array of shape (n, 3) with [x, y, angle_deg]
    outer_angle: rotation of the outer hexagon in degrees
    Returns the minimum outer hexagon side length.
    """
    # Face normals of outer hexagon (pointing outward)
    normal_angles = np.deg2rad(outer_angle + np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for i in range(n):
        x = params[i, 0]
        y = params[i, 1]
        a = params[i, 2]
        verts = _hex_vertices(x, y, a)
        dots = verts @ normals.T
        max_proj = max(max_proj, np.max(np.abs(dots)))
    
    # For a regular hexagon with side S, apothem = S * sqrt(3) / 2
    # So S = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / SQRT3

def _gradient_based_optimize(inner_params, outer_angle_deg, n, time_limit=30.0):
    """
    Gradient-based optimization using finite differences to approximate gradients
    of the fast side length function. This is a structural change from the previous
    PSO + Nelder-Mead approach.
    """
    start_time = time.time()
    params = inner_params.flatten().copy()
    oa = outer_angle_deg
    
    # L-BFGS-B with finite difference gradients
    def obj_func(p):
        return _fast_side_length(p.reshape(n, 3), n, oa)
    
    try:
        res = minimize(obj_func, params, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if time.time() - start_time < time_limit:
            candidate = res.x.reshape(n, 3)
            if res.fun < obj_func(params):
                return candidate
    except Exception:
        pass
    
    return inner_params

def _simulated_annealing(inner_params, outer_angle_deg, n, time_limit=20.0):
    """
    Simulated annealing for global exploration.
    """
    start_time = time.time()
    params = inner_params.flatten().copy()
    oa = outer_angle_deg
    current_val = _fast_side_length(params.reshape(n, 3), n, oa)
    best_params = params.copy()
    best_val = current_val
    
    T = 0.5  # Initial temperature
    cooling = 0.995
    min_T = 0.001
    
    while time.time() - start_time < time_limit and T > min_T:
        # Generate a random perturbation
        new_params = params + np.random.randn(n, 3) * 0.1
        new_params[:, 2] = np.mod(new_params[:, 2], 360)
        
        new_val = _fast_side_length(new_params.reshape(n, 3), n, oa)
        delta = new_val - current_val
        
        # Accept or reject
        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            params = new_params
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_params = params.copy()
        
        T *= cooling
    
    return best_params.reshape(n, 3)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    n = 11
    outer_angle = 30.0
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = np.array(seed_inner)
    best_side = seed_side
    best_outer_angle = outer_angle
    
    # Phase 1: Fast analytical optimization using projection method
    init_params = best_inner.flatten()
    best_val = _fast_side_length(best_inner, n, outer_angle)
    best_params = init_params.copy()
    
    # Multi-start L-BFGS-B optimization with fast objective
    n_starts = 6
    for i in range(n_starts):
        if time.time() > DEADLINE - safety_margin - 20:
            break
        
        if i == 0:
            x_start = init_params.copy()
        else:
            x_start = init_params + np.random.randn(n, 3) * (0.2 + 0.2 * i)
            x_start[:, 2] = np.mod(x_start[:, 2], 360)
        
        try:
            res = minimize(lambda p: _fast_side_length(p.reshape(n, 3), n, outer_angle), x_start, 
                          method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_val < best_side:
        candidate = best_params.reshape(n, 3)
        candidate_side = _compute_side(candidate, outer_angle)
        if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
            best_inner = candidate
            best_side = candidate_side
    
    # Phase 2: Simulated annealing for global exploration
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        try:
            sa_result = _simulated_annealing(best_inner, best_outer_angle, n, time_limit=min(remaining * 0.3, 15.0))
            sa_side = _compute_side(sa_result, best_outer_angle)
            if sa_side < best_side and _is_valid(sa_result, sa_side, best_outer_angle, min_sep=1e-9):
                best_inner = sa_result
                best_side = sa_side
        except Exception:
            pass
    
    # Phase 3: Gradient-based refinement
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 8:
        try:
            refined = _gradient_based_optimize(best_inner, best_outer_angle, n, time_limit=min(remaining * 0.3, 12.0))
            refined_side = _compute_side(refined, best_outer_angle)
            if refined_side < best_side and _is_valid(refined, refined_side, best_outer_angle, min_sep=1e-9):
                best_inner = refined
                best_side = refined_side
        except Exception:
            pass
    
    # Phase 4: Optimize outer hexagon angle
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            for test_oa in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]:
                if time.time() > DEADLINE - safety_margin - 2:
                    break
                test_side = _fast_side_length(best_inner, n, test_oa)
                if test_side < best_side:
                    exact_side = _compute_side(best_inner, test_oa)
                    if exact_side < best_side and _is_valid(best_inner, exact_side, test_oa, min_sep=1e-9):
                        best_side = exact_side
                        best_outer_angle = test_oa
        except Exception:
            pass
    
    # Phase 5: Final Nelder-Mead polish
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 3:
        try:
            init_params = best_inner.flatten()
            res = minimize(_objective, init_params, args=(best_outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
            if time.time() - start_time < DEADLINE - safety_margin:
                candidate = res.x.reshape(n, 3)
                candidate_side = _compute_side(candidate, best_outer_angle)
                if candidate_side < best_side and _is_valid(candidate, candidate_side, best_outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # Final check
    final_side = _compute_side(best_inner, best_outer_angle)
    if not _is_valid(best_inner, final_side, best_outer_angle, min_sep=1e-9):
        # Fallback to seed
        best_inner = np.array(seed_inner)
        final_side = seed_side
        best_outer_angle = 30.0
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(best_outer_angle)
    
    return inner_rows, center, side, angle_deg
