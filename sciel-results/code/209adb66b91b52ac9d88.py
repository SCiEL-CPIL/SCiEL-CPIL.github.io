import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3)

_BASE_ANGLES = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]))
_BASE_COS = np.cos(_BASE_ANGLES)
_BASE_SIN = np.sin(_BASE_ANGLES)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = _BASE_ANGLES + np.deg2rad(angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
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
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = np.clip(f / e, 0.0, 1.0)
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(c / a, 0.0, 1.0)
        return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = np.clip(s, 0.0, 1.0)
        t = (b * s + f) / e
        if t < 0:
            t = 0.0
            s = np.clip(c / a, 0.0, 1.0)
        elif t > 1:
            t = 1.0
            s = np.clip((b + f) / a, 0.0, 1.0)
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    t = np.clip(f / e, 0.0, 1.0)
    return np.linalg.norm(p1 - (p3 + t * d2))

def _poly_separation(p1, p2):
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    min_dist = float('inf')
    n1, n2 = len(p1), len(p2)
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
    angles = _BASE_ANGLES + np.deg2rad(angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _fast_side(inner_params, outer_angle_deg):
    """Vectorized support-function: minimum outer side for given inner config and outer angle."""
    n = inner_params.shape[0]
    xs = inner_params[:, 0]
    ys = inner_params[:, 1]
    angles_rad = np.deg2rad(inner_params[:, 2])
    cos_a = np.cos(angles_rad)[:, np.newaxis]
    sin_a = np.sin(angles_rad)[:, np.newaxis]
    vx = _BASE_COS * cos_a - _BASE_SIN * sin_a + xs[:, np.newaxis]
    vy = _BASE_SIN * cos_a + _BASE_COS * sin_a + ys[:, np.newaxis]
    normal_angles = np.deg2rad(30.0 + outer_angle_deg + np.arange(6) * 60.0)
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = vx[:, :, np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           vy[:, :, np.newaxis] * ny[np.newaxis, np.newaxis, :]
    max_per_hex = proj.max(axis=1)
    max_over_all = max_per_hex.max(axis=0)
    return 2.0 * float(np.max(max_over_all)) / math.sqrt(3.0)

def _active_vertex_info(inner_params, outer_angle_deg):
    """For each outer edge normal, find which inner hex/vertex gives the max projection."""
    n = inner_params.shape[0]
    xs = inner_params[:, 0]
    ys = inner_params[:, 1]
    angles_rad = np.deg2rad(inner_params[:, 2])
    cos_a = np.cos(angles_rad)[:, np.newaxis]
    sin_a = np.sin(angles_rad)[:, np.newaxis]
    vx = _BASE_COS * cos_a - _BASE_SIN * sin_a + xs[:, np.newaxis]
    vy = _BASE_SIN * cos_a + _BASE_COS * sin_a + ys[:, np.newaxis]
    normal_angles = np.deg2rad(30.0 + outer_angle_deg + np.arange(6) * 60.0)
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = vx[:, :, np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           vy[:, :, np.newaxis] * ny[np.newaxis, np.newaxis, :]
    # proj shape: (n, 6_vertices, 6_normals)
    # For each normal, find the (hex, vertex) pair with max projection
    max_idx = proj.argmax(axis=1)  # (n, 6_normals): which vertex of each hex is max for each normal
    proj_flat = proj.reshape(n * 6, 6)
    best_pair_per_normal = proj_flat.argmax(axis=0)  # (6,): flat index of best (hex,vertex) for each normal
    hex_idx = best_pair_per_normal // 6
    vert_idx = best_pair_per_normal % 6
    max_proj = proj_flat[best_pair_per_normal, np.arange(6)]
    side = 2.0 * float(np.max(max_proj)) / math.sqrt(3.0)
    return side, hex_idx, vert_idx, max_proj

def _overlap_penalty(inner_params, penalty_weight=50.0, threshold=0.005):
    n = inner_params.shape[0]
    centers = inner_params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < 2.5:
                v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
                v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
                sep = _poly_separation(v1, v2)
                if sep < threshold:
                    penalty += penalty_weight * (threshold - sep) ** 2
    return penalty

def _objective(params, outer_angle_deg, n):
    inner_params = params.reshape(n, 3)
    side = _fast_side(inner_params, outer_angle_deg)
    return side + _overlap_penalty(inner_params)

def _is_valid(inner_params, side, outer_angle_deg, min_sep=1e-8):
    n = inner_params.shape[0]
    side_check = _fast_side(inner_params, outer_angle_deg)
    if side_check > side + 1e-9:
        return False
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    for i in range(n):
        hex_v = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    centers = inner_params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < 2.5:
                v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
                v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
                sep = _poly_separation(v1, v2)
                if sep < min_sep:
                    return False
    return True

def _seed_solution():
    eps = 1e-9
    centers = [(0.0, 0.0)]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append(((D + eps) * np.cos(angle), (D + eps) * np.sin(angle)))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append(((3.0 + 2 * eps) * np.cos(angle), (3.0 + 2 * eps) * np.sin(angle)))
    inner = [[x, y, 30.0] for x, y in centers]
    arr = np.array(inner)
    best_side = float('inf')
    best_angle = 0.0
    for angle_deg in np.arange(0, 60, 1.0):
        s = _fast_side(arr, angle_deg)
        if s < best_side:
            best_side, best_angle = s, angle_deg
    for angle_deg in np.arange(best_angle - 1.0, best_angle + 1.0, 0.1):
        s = _fast_side(arr, angle_deg)
        if s < best_side:
            best_side, best_angle = s, angle_deg
    return inner, (0.0, 0.0), best_side, best_angle

def _lattice_packing():
    v1 = np.array([D, 0.0])
    v2 = np.array([D / 2, 1.5])
    selected = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (2,0), (0,-2)]
    inner = []
    for (i, j) in selected:
        pos = i * v1 + j * v2
        inner.append([float(pos[0]), float(pos[1]), 30.0])
    arr = np.array(inner)
    best_side, best_angle = float('inf'), 0.0
    for angle_deg in np.arange(0, 60, 0.5):
        s = _fast_side(arr, angle_deg)
        if s < best_side:
            best_side, best_angle = s, angle_deg
    for angle_deg in np.arange(best_angle - 0.5, best_angle + 0.5, 0.02):
        s = _fast_side(arr, angle_deg)
        if s < best_side:
            best_side, best_angle = s, angle_deg
    return inner, (0.0, 0.0), best_side, best_angle

def _simulated_annealing(inner_params, outer_angle_deg, time_limit, n=11):
    """
    Simulated annealing over positions and angles of inner hexagons.
    Uses the fast side computation as the energy, with overlap penalties.
    Key structural difference: explores the configuration space with random
    perturbations biased toward reducing the enclosing hexagon size.
    """
    current = inner_params.copy()
    current_side = _fast_side(current, outer_angle_deg)
    best = current.copy()
    best_side = current_side
    
    T = 0.5  # Initial temperature
    T_min = 1e-6
    cooling = 0.995
    rng = np.random.RandomState(12345)
    
    start = time.time()
    iters = 0
    
    while T > T_min and (time.time() - start) < time_limit:
        iters += 1
        
        # Generate perturbation: move one or two hexagons
        n_perturb = rng.randint(1, 3)
        candidates = rng.choice(n, size=n_perturb, replace=False)
        
        new_params = current.copy()
        for idx in candidates:
            # Random displacement
            dx = rng.randn() * 0.3
            dy = rng.randn() * 0.3
            da = rng.randn() * 5.0  # degrees
            new_params[idx, 0] += dx
            new_params[idx, 1] += dy
            new_params[idx, 2] += da
        
        new_side = _fast_side(new_params, outer_angle_deg)
        new_penalty = _overlap_penalty(new_params, penalty_weight=100.0, threshold=0.01)
        new_energy = new_side + new_penalty
        current_energy = current_side + _overlap_penalty(current, penalty_weight=100.0, threshold=0.01)
        
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current = new_params
            current_side = new_side
            if current_side < best_side - 1e-12:
                best = current.copy()
                best_side = current_side
        
        T *= cooling
        
        # Periodic restarts from best to avoid getting stuck
        if iters % 500 == 0:
            current = best.copy()
            current_side = best_side
    
    return best, best_side

def _active_set_optimize(inner_params, outer_angle_deg, time_limit, n=11):
    """
    Active-set optimization: identify which vertices constrain the outer hexagon,
    then optimize positions to minimize the maximum projection while maintaining
    separation constraints.
    
    The key insight: the outer side is determined by max over 6 edge normals.
    We can formulate this as: minimize max_k (max over active vertices of proj_k)
    subject to separation constraints.
    """
    current = inner_params.copy()
    current_side = _fast_side(current, outer_angle_deg)
    best = current.copy()
    best_side = current_side
    
    start = time.time()
    
    # Iteratively refine: alternate between optimizing positions given active set
    # and updating the active set
    for iteration in range(50):
        if time.time() > time_limit:
            break
        
        # Get current active vertices
        side, hex_idx, vert_idx, max_proj = _active_vertex_info(current, outer_angle_deg)
        
        # Objective: minimize the maximum projection
        # Use SLSQP with the constraint that each active vertex's projection
        # is bounded
        
        def obj(params):
            p = params.reshape(n, 3)
            return _fast_side(p, outer_angle_deg)
        
        # Try optimizing with L-BFGS-B, which handles the smooth max well
        try:
            res = minimize(obj, current.flatten(), method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000})
            
            if res.fun < current_side - 1e-12:
                new_params = res.x.reshape(n, 3)
                # Check validity
                new_side = _fast_side(new_params, outer_angle_deg)
                if _is_valid(new_params, new_side, outer_angle_deg, min_sep=1e-7):
                    current = new_params
                    current_side = new_side
                    if current_side < best_side - 1e-12:
                        best = current.copy()
                        best_side = current_side
        except Exception:
            pass
        
        # Try Nelder-Mead for local refinement
        if time.time() < time_limit - 5:
            try:
                res2 = minimize(obj, current.flatten(), method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res2.fun < current_side - 1e-12:
                    new_params = res2.x.reshape(n, 3)
                    new_side = _fast_side(new_params, outer_angle_deg)
                    if _is_valid(new_params, new_side, outer_angle_deg, min_sep=1e-7):
                        current = new_params
                        current_side = new_side
                        if current_side < best_side - 1e-12:
                            best = current.copy()
                            best_side = current_side
            except Exception:
                pass
    
    return best, best_side

def _polish_angle(inner_arr, initial_angle, time_limit):
    """High-precision polish of the outer hexagon angle."""
    best_side = _fast_side(inner_arr, initial_angle)
    best_angle = initial_angle
    
    scales = [
        (1.0, 10.0),
        (0.1, 1.0),
        (0.01, 0.1),
        (0.001, 0.01),
    ]
    
    for step, span in scales:
        if time.time() > time_limit:
            break
        angles = np.arange(best_angle - span, best_angle + span + step/2, step)
        for angle_deg in angles:
            if time.time() > time_limit:
                break
            side = _fast_side(inner_arr, angle_deg)
            if side < best_side - 1e-15:
                best_side, best_angle = side, angle_deg
    
    if time.time() < time_limit:
        try:
            res = minimize_scalar(
                lambda a: _fast_side(inner_arr, a),
                bounds=(best_angle - 0.001, best_angle + 0.001),
                method='bounded', options={'xatol': 1e-14}
            )
            if res.success and res.fun < best_side - 1e-15:
                best_side, best_angle = res.fun, res.x
        except Exception:
            pass
    
    return best_side, best_angle

def _rigid_rotation_optimize(base_params, time_limit):
    """Rigid body optimization: rotate and translate the whole cluster."""
    base = base_params.copy()
    base_angles = base[:, 2].copy()

    def obj(p):
        if time.time() > time_limit:
            return 1e9
        ga = p[0]
        tx, ty = p[1], p[2]
        new_params = base.copy()
        new_params[:, 0] = base[:, 0] + tx
        new_params[:, 1] = base[:, 1] + ty
        new_params[:, 2] = base_angles + ga
        return _fast_side(new_params, 0.0)

    best_val = obj(np.array([0.0, 0.0, 0.0]))
    best_p = np.array([0.0, 0.0, 0.0])

    for ga in np.arange(0, 60, 0.5):
        val = obj(np.array([ga, 0.0, 0.0]))
        if val < best_val:
            best_val, best_p = val, np.array([ga, 0.0, 0.0])
        if time.time() > time_limit - 5:
            break

    if time.time() < time_limit - 5:
        res = minimize_scalar(
            lambda ga: obj(np.array([ga, best_p[1], best_p[2]])),
            bounds=(best_p[0] - 0.5, best_p[0] + 0.5),
            method='bounded', options={'xatol': 1e-8}
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_p[0] = res.x

    if time.time() < time_limit - 5:
        res = minimize(
            lambda t: obj(np.array([best_p[0], t[0], t[1]])),
            best_p[1:3], method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_p[1:3] = res.x

    if time.time() < time_limit - 3:
        res = minimize(obj, best_p, method='Nelder-Mead',
                       options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_p = res.x

    ga, tx, ty = best_p
    new_params = base.copy()
    new_params[:, 0] = base[:, 0] + tx
    new_params[:, 1] = base[:, 1] + ty
    new_params[:, 2] = base_angles + ga
    return new_params, best_val

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    n = 11

    # Generate seed candidates
    seed1_inner, _, seed1_side, seed1_angle = _seed_solution()
    seed2_inner, _, seed2_side, seed2_angle = _lattice_packing()
    seed_candidates = [(seed1_inner, seed1_side, seed1_angle),
                       (seed2_inner, seed2_side, seed2_angle)]
    best_seed = min(seed_candidates, key=lambda x: x[1])
    best_inner = np.array(best_seed[0], dtype=float)
    best_side = best_seed[1]
    best_angle = best_seed[2]

    # Phase 1: Rigid-body rotation optimization
    if time.time() < DEADLINE - safety_margin - 30:
        try:
            rotated, side_val = _rigid_rotation_optimize(best_inner, DEADLINE - safety_margin - 20)
            if _is_valid(rotated, side_val, 0.0, min_sep=1e-8):
                best_inner, best_side, best_angle = rotated, side_val, 0.0
        except Exception:
            pass

    # Phase 2: Simulated annealing (STRUCTURAL CHANGE - different search paradigm)
    if time.time() < DEADLINE - safety_margin - 25:
        try:
            sa_inner, sa_side = _simulated_annealing(
                best_inner, best_angle, DEADLINE - safety_margin - 15, n=n
            )
            if sa_side < best_side - 1e-10:
                best_inner, best_side = sa_inner, sa_side
        except Exception:
            pass

    # Phase 3: Active-set optimization (STRUCTURAL CHANGE - exploits problem structure)
    if time.time() < DEADLINE - safety_margin - 15:
        try:
            as_inner, as_side = _active_set_optimize(
                best_inner, best_angle, DEADLINE - safety_margin - 5, n=n
            )
            if as_side < best_side - 1e-10:
                best_inner, best_side = as_inner, as_side
        except Exception:
            pass

    # Phase 4: Second round of simulated annealing from improved starting point
    if time.time() < DEADLINE - safety_margin - 10:
        try:
            sa2_inner, sa2_side = _simulated_annealing(
                best_inner, best_angle, DEADLINE - safety_margin - 5, n=n
            )
            if sa2_side < best_side - 1e-10:
                best_inner, best_side = sa2_inner, sa2_side
        except Exception:
            pass

    # Phase 5: High-precision angle polish
    if time.time() < DEADLINE - safety_margin:
        best_side, best_angle = _polish_angle(best_inner, best_angle, DEADLINE - safety_margin)

    # Phase 6: Final local refinement
    if time.time() < DEADLINE - safety_margin - 3:
        init_params = best_inner.flatten()
        try:
            res = minimize(_objective, init_params, args=(best_angle, n), method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if time.time() < DEADLINE - safety_margin:
                cand = res.x.reshape(n, 3)
                cand_side = _fast_side(cand, best_angle)
                if _is_valid(cand, cand_side, best_angle, min_sep=1e-8) and cand_side < best_side - 1e-12:
                    best_inner, best_side = cand, cand_side
        except Exception:
            pass

    # Phase 7: Final angle polish
    if time.time() < DEADLINE - safety_margin:
        best_side, best_angle = _polish_angle(best_inner, best_angle, DEADLINE - safety_margin)

    # Final validation
    final_side = _fast_side(best_inner, best_angle)
    if not _is_valid(best_inner, final_side, best_angle, min_sep=1e-8):
        best_inner = np.array(best_seed[0], dtype=float)
        best_side, best_angle = best_seed[1], best_seed[2]
    else:
        best_side = final_side

    inner_rows = [[float(best_inner[i, 0]), float(best_inner[i, 1]), float(best_inner[i, 2])]
                  for i in range(n)]
    return inner_rows, (0.0, 0.0), float(best_side), float(best_angle)
