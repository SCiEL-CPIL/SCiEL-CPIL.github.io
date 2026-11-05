import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def _soft_max(vals, beta=50.0):
    """Smooth approximation of max via log-sum-exp."""
    m = np.max(vals)
    if m < -1e10:
        return m
    return m + np.log(np.sum(np.exp(vals - m))) / beta

def _support_function_smooth(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length using smooth support function.
    """
    n = len(inner_params)
    outer_angle_rad = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle_rad + np.deg2rad(30 + np.array([0, 60, 120, 180, 240, 300]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    all_pts = []
    for i in range(n):
        x, y, a = inner_params[i]
        vs = _hex_vertices(x, y, a)
        all_pts.append(vs)
    all_pts = np.vstack(all_pts)
    
    dots = normals @ all_pts.T
    max_dots = np.max(dots, axis=1)
    
    sm = _soft_max(max_dots, beta=100.0)
    
    return sm / (SQRT3 / 2.0)

def _objective_smooth(params, outer_angle_deg, n):
    """Smooth objective: soft support function + overlap penalty."""
    inner_params = params.reshape(n, 3)
    side = _support_function_smooth(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

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
    """Compute the minimum distance between two convex polygons. Returns 0 if they overlap."""
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
    """
    n = len(inner_params)
    
    all_pts = []
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        all_pts.append(hex_v)
    all_pts = np.vstack(all_pts)
    
    outer_angle_rad = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle_rad + np.deg2rad(30 + np.array([0, 60, 120, 180, 240, 300]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    dots = normals @ all_pts.T
    max_dots = np.max(dots, axis=1)
    
    return np.max(max_dots) / (SQRT3 / 2.0)

def _overlap_penalty(inner_params, penalty_weight=100.0):
    """Compute penalty for overlapping inner hexagons."""
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
    """Objective function: outer side length + overlap penalty."""
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
    """Return the concentric ring seed solution."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (SQRT3 + eps) * np.cos(angle)
        y = (SQRT3 + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    max_x = (2 + eps) * SQRT3
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / SQRT3
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

def _symmetric_seed_params():
    """
    Generate a symmetric configuration based on 6 parameters:
    p[0]: r1 (distance of inner ring)
    p[1]: r2 (distance of outer ring)
    p[2]: theta1 (rotation of inner ring)
    p[3]: theta2 (rotation of outer ring)
    p[4]: alpha1 (rotation of inner hexes)
    p[5]: alpha2 (rotation of outer hexes)
    """
    r1, r2, theta1, theta2, alpha1, alpha2 = p
    params = np.zeros((11, 3))
    
    # Center
    params[0] = [0.0, 0.0, 30.0]
    
    # Inner ring (6 hexes)
    for k in range(6):
        angle = np.deg2rad(theta1 + k * 60)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        params[k+1] = [x, y, alpha1]
        
    # Outer ring (4 hexes)
    # Placed at 90 degree intervals, rotated by theta2
    for k in range(4):
        angle = np.deg2rad(theta2 + k * 90)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        params[k+7] = [x, y, alpha2]
        
    return params

def _optimize_symmetric():
    """
    Optimize the symmetric configuration.
    """
    # Initial guess:
    # r1 = sqrt(3) ~ 1.732
    # r2 = 3.0
    # theta1 = 0
    # theta2 = 30
    # alpha1 = 30
    # alpha2 = 30
    x0 = np.array([SQRT3, 3.0, 0.0, 30.0, 30.0, 30.0])
    
    def obj_sym(params):
        inner_params = _symmetric_seed_params(params)
        side = _compute_side(inner_params, 30.0)
        pen = _overlap_penalty(inner_params)
        return side + pen
    
    # Use Powell method which is robust for non-smooth functions
    res = minimize(obj_sym, x0, method='Powell', options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8})
    
    best_params = res.x
    best_inner = _symmetric_seed_params(best_params)
    best_side = _compute_side(best_inner, 30.0)
    
    return best_inner, best_side

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    n = 11
    outer_angle = 30.0
    
    # 1. Try Symmetric Optimization
    best_inner = None
    best_side = float('inf')
    
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        try:
            sym_inner, sym_side = _optimize_symmetric()
            if sym_side < best_side:
                best_inner = sym_inner
                best_side = sym_side
        except Exception as e:
            pass

    # 2. Fallback to Seed if symmetric failed or is worse
    if best_inner is None:
        seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
        best_inner = np.array(seed_inner)
        best_side = seed_side
    else:
        # Compare with seed
        seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
        if seed_side < best_side:
            best_inner = np.array(seed_inner)
            best_side = seed_side

    # 3. Local Refinement on the best candidate
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        try:
            # Use Nelder-Mead on the full 33 parameters for a few steps
            # to break symmetry if beneficial
            init_params = best_inner.flatten()
            
            # We need to be careful with time. Nelder-Mead on 33 vars is slow.
            # Limit iterations.
            res = minimize(_objective, init_params, args=(outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
            
            candidate = res.x.reshape(n, 3)
            cand_side = _compute_side(candidate, outer_angle)
            
            if cand_side < best_side and _is_valid(candidate, cand_side, outer_angle, min_sep=1e-6):
                best_inner = candidate
                best_side = cand_side
        except Exception:
            pass

    # 4. Final Polish with tighter tolerance
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            init_params = best_inner.flatten()
            res = minimize(_objective, init_params, args=(outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            
            candidate = res.x.reshape(n, 3)
            cand_side = _compute_side(candidate, outer_angle)
            
            if cand_side < best_side and _is_valid(candidate, cand_side, outer_angle, min_sep=1e-8):
                best_inner = candidate
                best_side = cand_side
        except Exception:
            pass

    # Final check
    final_side = _compute_side(best_inner, outer_angle)
    if not _is_valid(best_inner, final_side, outer_angle, min_sep=1e-8):
        # Fallback to seed
        seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
        best_inner = np.array(seed_inner)
        final_side = seed_side
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(outer_angle)
    
    return inner_rows, center, side, angle_deg
