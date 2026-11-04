import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3)

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
    Returns 0 if they overlap (interiors intersect or share boundary),
    positive if separated.
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

def _min_enclosing_hex_side(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon, given a fixed outer angle.
    """
    n = len(inner_params)
    apothem_factor = math.sqrt(3) / 2.0
    
    normal_angles = np.deg2rad(outer_angle_deg + 30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_val = 0.0
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            projections = normals @ v
            for proj in projections:
                if proj > 0:
                    required_s = proj / apothem_factor
                    if required_s > max_val:
                        max_val = required_s
    
    return max_val

def _compute_side(inner_params, outer_angle_deg):
    """Compute minimum side length for given inner params and outer angle."""
    return _min_enclosing_hex_side(inner_params, outer_angle_deg)

def _overlap_penalty(inner_params, penalty_weight=50.0):
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
            if sep < 0.005:
                penalty += penalty_weight * (0.005 - sep) ** 2
    
    return penalty

def _objective(params, outer_angle_deg, n):
    """Objective: outer side length + overlap penalty."""
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if configuration is valid."""
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
    """Concentric ring seed solution with exact optimal positions."""
    sqrt3 = math.sqrt(3)
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # Compute exact minimum side
    inner_arr = np.array(inner)
    best_side = float('inf')
    best_angle = 0.0
    for angle_deg in np.arange(0, 60, 0.5):
        side = _compute_side(inner_arr, angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    for angle_deg in np.arange(best_angle - 1, best_angle + 1, 0.01):
        side = _compute_side(inner_arr, angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    return inner, (0.0, 0.0), best_side, best_angle

def _lattice_packing():
    """
    Generate a hexagonal lattice packing of 11 unit hexagons.
    """
    sqrt3 = math.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    selected = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (2,0), (0,-2)]
    
    centers = []
    for (i, j) in selected:
        pos = i * v1 + j * v2
        centers.append(pos)
    
    inner = []
    for pos in centers:
        inner.append([float(pos[0]), float(pos[1]), 30.0])
    
    best_side = float('inf')
    best_angle = 0.0
    for angle_deg in np.arange(0, 60, 1.0):
        side = _compute_side(np.array(inner), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    for angle_deg in np.arange(best_angle - 2, best_angle + 2, 0.1):
        side = _compute_side(np.array(inner), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    return inner, (0.0, 0.0), best_side, best_angle

def _polish_angle(inner_arr, initial_angle, time_limit):
    """High-precision polish of the outer hexagon angle."""
    best_side = _compute_side(inner_arr, initial_angle)
    best_angle = initial_angle
    
    for angle_deg in np.arange(initial_angle - 5, initial_angle + 5, 0.5):
        if time.time() > time_limit:
            break
        side = _compute_side(inner_arr, angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    for angle_deg in np.arange(best_angle - 0.5, best_angle + 0.5, 0.05):
        if time.time() > time_limit:
            break
        side = _compute_side(inner_arr, angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    if time.time() < time_limit:
        try:
            res = minimize_scalar(
                lambda a: _compute_side(inner_arr, a),
                bounds=(best_angle - 0.1, best_angle + 0.1),
                method='bounded',
                options={'xatol': 1e-10}
            )
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_angle = res.x
        except Exception:
            pass
    
    return best_side, best_angle

def _direct_optimize_positions(inner_seed, outer_angle, time_limit):
    """Directly optimize inner positions using coordinate descent."""
    n = len(inner_seed)
    inner_arr = np.array(inner_seed, dtype=float)
    
    # Coordinate descent: optimize each hexagon's position and angle
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_limit:
                break
            
            # Save current position
            old_x, old_y, old_a = inner_arr[i].copy()
            
            # Try small perturbations
            best_x, best_y, best_a = old_x, old_y, old_a
            best_side = _compute_side(inner_arr, outer_angle)
            
            # Perturb position
            for dx in [-0.1, -0.05, 0, 0.05, 0.1]:
                for dy in [-0.1, -0.05, 0, 0.05, 0.1]:
                    inner_arr[i, 0] = old_x + dx
                    inner_arr[i, 1] = old_y + dy
                    side = _compute_side(inner_arr, outer_angle)
                    if side < best_side - 1e-8:
                        best_side = side
                        best_x, best_y = old_x + dx, old_y + dy
                        improved = True
            
            # Perturb angle
            for da in [-10, -5, 0, 5, 10]:
                inner_arr[i, 0] = best_x
                inner_arr[i, 1] = best_y
                inner_arr[i, 2] = old_a + da
                side = _compute_side(inner_arr, outer_angle)
                if side < best_side - 1e-8:
                    best_side = side
                    best_a = old_a + da
                    improved = True
            
            inner_arr[i] = [best_x, best_y, best_a]
        
        if not improved:
            break
    
    return inner_arr

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    n = 11
    
    # Try multiple seed configurations
    seed_candidates = []
    
    seed1_inner, seed1_center, seed1_side, seed1_angle = _seed_solution()
    seed_candidates.append((seed1_inner, seed1_side, seed1_angle))
    
    seed2_inner, seed2_center, seed2_side, seed2_angle = _lattice_packing()
    seed_candidates.append((seed2_inner, seed2_side, seed2_angle))
    
    best_seed = min(seed_candidates, key=lambda x: x[1])
    best_inner = [list(row) for row in best_seed[0]]
    best_side = best_seed[1]
    best_angle = best_seed[2]
    
    # Phase 1: Direct coordinate descent optimization
    if time.time() < DEADLINE - safety_margin - 20:
        inner_arr = np.array(best_inner)
        optimized = _direct_optimize_positions(inner_arr, best_angle, DEADLINE - safety_margin - 20)
        candidate_side = _compute_side(optimized, best_angle)
        if _is_valid(optimized, candidate_side, best_angle, min_sep=1e-8):
            if candidate_side < best_side:
                best_inner = optimized.tolist()
                best_side = candidate_side
    
    # Phase 2: Nelder-Mead optimization for multiple angles
    outer_angles_to_try = [best_angle, 0.0, 15.0, 30.0, 45.0]
    
    for outer_angle in outer_angles_to_try:
        if time.time() > DEADLINE - safety_margin - 30:
            break
        
        try:
            init_params = np.zeros(3 * n)
            for i, (x, y, a) in enumerate(best_inner):
                init_params[i*3] = x
                init_params[i*3+1] = y
                init_params[i*3+2] = a
            
            res = minimize(_objective, init_params.copy(), args=(outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            
            if time.time() > DEADLINE - safety_margin:
                break
            
            candidate_inner = res.x.reshape(n, 3)
            candidate_side = _compute_side(candidate_inner, outer_angle)
            
            if _is_valid(candidate_inner, candidate_side, outer_angle, min_sep=1e-8):
                if candidate_side < best_side:
                    best_inner = candidate_inner.tolist()
                    best_side = candidate_side
                    best_angle = outer_angle
        except Exception:
            pass
    
    # Phase 3: High-precision angle polish
    if time.time() < DEADLINE - safety_margin:
        inner_arr = np.array(best_inner)
        best_side, best_angle = _polish_angle(inner_arr, best_angle, DEADLINE - safety_margin)
    
    # Phase 4: Final coordinate descent with refined angle
    if time.time() < DEADLINE - safety_margin - 10:
        inner_arr = np.array(best_inner)
        optimized = _direct_optimize_positions(inner_arr, best_angle, DEADLINE - safety_margin - 10)
        candidate_side = _compute_side(optimized, best_angle)
        if _is_valid(optimized, candidate_side, best_angle, min_sep=1e-8):
            if candidate_side < best_side:
                best_inner = optimized.tolist()
                best_side = candidate_side
    
    # Phase 5: Final high-precision angle polish
    if time.time() < DEADLINE - safety_margin:
        inner_arr = np.array(best_inner)
        best_side, best_angle = _polish_angle(inner_arr, best_angle, DEADLINE - safety_margin)
    
    # Final validation and safety adjustment
    inner_arr_final = np.array(best_inner)
    final_side = _compute_side(inner_arr_final, best_angle)
    if not _is_valid(inner_arr_final, final_side, best_angle, min_sep=1e-8):
        best_inner = [list(row) for row in best_seed[0]]
        best_side = best_seed[1]
        best_angle = best_seed[2]
        final_side = best_side
    else:
        best_side = final_side
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
    
    center = (0.0, 0.0)
    side = float(best_side)
    angle_deg = float(best_angle)
    
    return inner_rows, center, side, angle_deg
