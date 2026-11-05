import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import ConvexHull

N = 11
SQRT3 = math.sqrt(3)

def _soft_max(values, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(values)
    if m == -np.inf:
        return -np.inf
    return m + math.log(np.sum(np.exp(beta * (values - m)))) / beta

def _smooth_side(inner_params, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for the outer hexagon side length using soft-max.
    Returns a differentiable approximation of the true side length.
    """
    n = len(inner_params)
    normals = np.array([
        [np.cos(np.deg2rad(outer_angle_deg + 30 + k * 60)), 
         np.sin(np.deg2rad(outer_angle_deg + 30 + k * 60))]
        for k in range(6)
    ])
    
    max_proj = 0.0
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        verts = _hex_vertices(x, y, a)
        dots = verts @ normals.T
        max_per_normal = np.max(dots, axis=0)
        # Use soft-max over the 6 normals for smoothness
        sm = _soft_max(max_per_normal, beta)
        if sm > max_proj:
            max_proj = sm
            
    return 2.0 * max_proj / SQRT3

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

def _poly_separation_fast(p1, p2):
    """Fast separation check with bounding box early exit."""
    min1 = np.min(p1, axis=0)
    max1 = np.max(p1, axis=0)
    min2 = np.min(p2, axis=0)
    max2 = np.max(p2, axis=0)
    
    # Bounding box check
    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        dx = max(min2[0] - max1[0], max1[0] - max2[0], 0.0)
        dy = max(min2[1] - max1[1], max1[1] - max2[1], 0.0)
        return math.sqrt(dx*dx + dy*dy)
        
    # Check vertex in polygon
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
            
    # Check edge intersections
    n1 = len(p1)
    n2 = len(p2)
    min_dist = float('inf')
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

def _fast_min_side(inner_params, outer_angle_deg):
    """
    Compute the minimum side length of the outer hexagon (fixed orientation)
    that contains all inner hexagons.
    """
    n = len(inner_params)
    normals = np.array([
        [np.cos(np.deg2rad(outer_angle_deg + 30 + k * 60)), 
         np.sin(np.deg2rad(outer_angle_deg + 30 + k * 60))]
        for k in range(6)
    ])
    
    max_proj = 0.0
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        verts = _hex_vertices(x, y, a)
        dots = verts @ normals.T
        max_per_normal = np.max(dots, axis=0)
        current_max = np.max(max_per_normal)
        if current_max > max_proj:
            max_proj = current_max
            
    return 2.0 * max_proj / SQRT3

def _find_optimal_outer_angle(inner_params):
    """
    Find the optimal rotation angle for the outer hexagon
    given a fixed configuration of inner hexagons.
    """
    # Collect all vertices
    all_verts = []
    for i in range(len(inner_params)):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        verts = _hex_vertices(x, y, a)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    
    # Compute Convex Hull
    try:
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[hull.vertices]
    except Exception:
        # Fallback to all vertices if hull fails (e.g., degenerate)
        hull_verts = all_verts
        
    # Objective function for 1D search
    def side_length_for_angle(alpha_deg):
        normals = np.array([
            [np.cos(np.deg2rad(alpha_deg + 30 + k * 60)), 
             np.sin(np.deg2rad(alpha_deg + 30 + k * 60))]
            for k in range(6)
        ])
        # Project hull vertices onto normals
        dots = hull_verts @ normals.T
        max_proj = np.max(dots)
        return 2.0 * max_proj / SQRT3
        
    # Grid search to find approximate minimum
    alphas = np.linspace(0, 60, 121)
    sides = [side_length_for_angle(a) for a in alphas]
    best_idx = np.argmin(sides)
    best_alpha_approx = alphas[best_idx]
    
    # Refine with minimize_scalar
    # Bound the search to a small window around the grid minimum
    lower = max(0, best_alpha_approx - 3)
    upper = min(60, best_alpha_approx + 3)
    
    try:
        res = minimize_scalar(side_length_for_angle, bounds=(lower, upper), method='bounded', options={'xatol': 1e-10})
        return res.x
    except Exception:
        return best_alpha_approx

def _coordinate_descent_polish(params, outer_angle_deg, n, max_iters=50, tol=1e-12, time_budget=30.0):
    """
    Coordinate descent with adaptive-window golden-section search on each parameter.
    Uses the true (non-smooth) side length for evaluation.
    """
    start = time.time()
    params = params.copy()
    current_val = _fast_min_side(params.reshape(n, 3), outer_angle_deg)
    
    # Adaptive window sizes per parameter type
    # x, y: spatial coordinates; angle: rotation
    window_sizes = [0.05, 0.05, 5.0]  # smaller windows for faster convergence
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(n):
            for j in range(3):
                # Golden-section search on parameter (i, j)
                p = params[i * 3 + j]
                window = window_sizes[j] * (1.0 - 0.3 * it / max_iters)  # shrink over time
                a = p - window
                b = p + window
                
                gr = (math.sqrt(5) - 1) / 2
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                
                # Evaluate at c and d
                params_c = params.copy()
                params_c[i * 3 + j] = c
                val_c = _fast_min_side(params_c.reshape(n, 3), outer_angle_deg)
                
                params_d = params.copy()
                params_d[i * 3 + j] = d
                val_d = _fast_min_side(params_d.reshape(n, 3), outer_angle_deg)
                
                # Iterate golden section - more iterations for precision
                for _ in range(40):
                    if val_c < val_d:
                        b = d
                        d = c
                        val_d = val_c
                        c = b - gr * (b - a)
                        params_c = params.copy()
                        params_c[i * 3 + j] = c
                        val_c = _fast_min_side(params_c.reshape(n, 3), outer_angle_deg)
                    else:
                        a = c
                        c = d
                        val_c = val_d
                        d = a + gr * (b - a)
                        params_d = params.copy()
                        params_d[i * 3 + j] = d
                        val_d = _fast_min_side(params_d.reshape(n, 3), outer_angle_deg)
                
                # Best is min of c and d
                if val_c < val_d:
                    best_p = c
                    best_val = val_c
                else:
                    best_p = d
                    best_val = val_d
                
                if abs(best_p - p) > tol and best_val < current_val - 1e-15:
                    params[i * 3 + j] = best_p
                    current_val = best_val
                    improved = True
        
        if not improved:
            break
    
    return params, current_val

def _overlap_penalty(inner_params, penalty_weight=200.0):
    """
    Compute penalty for overlapping inner hexagons.
    Uses center-distance early exit for speed.
    """
    n = len(inner_params)
    penalty = 0.0
    # Minimum center distance for two unit hexagons to be disjoint is 2*sqrt(3)
    # (distance between centers when edges touch)
    min_center_dist = 2.0 * SQRT3 - 0.01  # small tolerance
    
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            
            # Quick center distance check
            dx = x2 - x1
            dy = y2 - y1
            center_dist_sq = dx*dx + dy*dy
            
            # If centers are far enough apart, no overlap possible
            # Max distance from center to vertex is 1.0, so if center_dist > 2.0, definitely no overlap
            if center_dist_sq > 4.1:  # (2.05)^2
                continue
            
            v1 = _hex_vertices(x1, y1, a1)
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation_fast(v1, v2)
            if sep < 0.01:
                penalty += penalty_weight * (0.01 - sep) ** 2
    
    return penalty

def _objective(params, outer_angle_deg, n):
    """
    Objective function: outer side length + overlap penalty.
    Uses fast side length calculation.
    """
    inner_params = params.reshape(n, 3)
    side = _fast_min_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if the configuration is valid."""
    n = len(inner_params)
    
    # Check containment using fast side length
    # The side length must be at least the fast_min_side
    # Actually, we just need to check if the given side is sufficient.
    # But _is_valid is called with a specific side.
    # We can check if _fast_min_side <= side + epsilon.
    # And check disjointness.
    
    fast_side = _fast_min_side(inner_params, outer_angle_deg)
    if fast_side > side + 1e-9:
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
            
            sep = _poly_separation_fast(v1, v2)
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
    
    # Calculate exact side for seed
    inner_arr = np.array(inner)
    side = _fast_min_side(inner_arr, 30.0)
    
    return inner, (0.0, 0.0), side, 30.0

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    n = 11
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = np.array(seed_inner)
    
    # Phase 0: Optimize outer angle for the seed shape (1D search)
    optimal_angle = _find_optimal_outer_angle(best_inner)
    
    # Update best side with optimal angle
    best_side = _fast_min_side(best_inner, optimal_angle)
    outer_angle = optimal_angle
    
    init_params = best_inner.flatten()
    
    # Phase 1: Powell optimization with fast objective
    # Powell is good for this kind of problem.
    best_params = init_params.copy()
    best_val = _objective(best_params, outer_angle, n)
    
    # Multi-start Powell
    n_starts = 12
    for i in range(n_starts):
        if time.time() > DEADLINE - safety_margin - 25:
            break
        
        if i == 0:
            x_start = init_params.copy()
        else:
            # Larger perturbations for later starts to explore more
            scale = 0.05 * (1 + i * 0.3)
            x_start = init_params + np.random.randn(n, 3) * scale
            x_start[:, 2] = np.mod(x_start[:, 2], 360)
        
        try:
            res = minimize(_objective, x_start, args=(outer_angle, n), 
                          method='Powell',
                          options={'maxiter': 3000, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # After Powell, re-optimize outer angle for the best configuration
    if best_val < best_side:
        candidate = best_params.reshape(n, 3)
        optimal_angle_candidate = _find_optimal_outer_angle(candidate)
        candidate_side = _fast_min_side(candidate, optimal_angle_candidate)
        if _is_valid(candidate, candidate_side, optimal_angle_candidate, min_sep=1e-9):
            best_inner = candidate
            best_side = candidate_side
            outer_angle = optimal_angle_candidate
            
    # Phase 2: Nelder-Mead refinement
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            res = minimize(_objective, best_inner.flatten(), args=(outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_side:
                candidate = res.x.reshape(n, 3)
                # Re-optimize outer angle after NM refinement
                opt_angle_nm = _find_optimal_outer_angle(candidate)
                candidate_side = _fast_min_side(candidate, opt_angle_nm)
                if _is_valid(candidate, candidate_side, opt_angle_nm, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
                    outer_angle = opt_angle_nm
        except Exception:
            pass
            
    # Phase 3: Coordinate descent polish with golden-section search
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        try:
            polished_params, polished_val = _coordinate_descent_polish(
                best_inner.flatten(), outer_angle, n, 
                max_iters=50, tol=1e-15, time_budget=min(remaining - 5, 50.0))
            if polished_val < best_side:
                candidate = polished_params.reshape(n, 3)
                # Re-optimize outer angle after coordinate descent
                opt_angle_cd = _find_optimal_outer_angle(candidate)
                candidate_side = _fast_min_side(candidate, opt_angle_cd)
                if _is_valid(candidate, candidate_side, opt_angle_cd, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
                    outer_angle = opt_angle_cd
        except Exception:
            pass
    
    # Phase 4: Final tight Nelder-Mead polish
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 3:
        try:
            res = minimize(_objective, best_inner.flatten(), args=(outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-15})
            if res.fun < best_side:
                candidate = res.x.reshape(n, 3)
                # Final outer angle optimization
                opt_angle_final = _find_optimal_outer_angle(candidate)
                candidate_side = _fast_min_side(candidate, opt_angle_final)
                if _is_valid(candidate, candidate_side, opt_angle_final, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
                    outer_angle = opt_angle_final
        except Exception:
            pass
            
    # Final check - one more outer angle optimization for the final configuration
    final_opt_angle = _find_optimal_outer_angle(best_inner)
    final_side = _fast_min_side(best_inner, final_opt_angle)
    if not _is_valid(best_inner, final_side, final_opt_angle, min_sep=1e-9):
        # Fallback to seed
        best_inner = np.array(seed_inner)
        final_side = seed_side
        outer_angle = seed_angle
    else:
        outer_angle = final_opt_angle
        
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(outer_angle)
    
    return inner_rows, center, side, angle_deg
