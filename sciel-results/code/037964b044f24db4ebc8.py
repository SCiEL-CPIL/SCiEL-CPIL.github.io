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

def _smooth_side(inner_params, outer_angle_deg, beta=50.0):
    """Smooth approximation of the support function using log-sum-exp."""
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
    proj_flat = proj.reshape(n * 6, 6)
    max_proj = np.max(proj_flat, axis=0)
    shifted = proj_flat - max_proj
    se = np.sum(np.exp(shifted), axis=0)
    smooth_max_proj = np.log(se) + max_proj
    m2 = np.max(smooth_max_proj)
    se2 = np.sum(np.exp(smooth_max_proj - m2))
    final_smooth_max = np.log(se2) + m2
    return 2.0 * float(final_smooth_max) / math.sqrt(3.0)

def _support_function(phi, theta):
    """
    Calculate the support function h(phi, theta) of a unit regular hexagon
    rotated by theta (radians). phi is the direction angle (radians).
    """
    # theta is rotation of the hexagon.
    # The hexagon has vertices at angles theta + k*pi/3.
    # The normals to the faces are at theta + pi/6 + k*pi/3.
    # We want the distance from center to the boundary in direction phi.
    # This is the distance to the face whose normal is closest to phi.
    
    # Relative angle to the nearest face normal
    # Face normals are at theta + pi/6 + k*pi/3
    # Let delta = phi - (theta + pi/6)
    # We want delta modulo pi/3, centered at 0.
    
    delta = (phi - (theta + math.pi / 6.0)) % (math.pi / 3.0)
    # Center in [-pi/6, pi/6]
    delta = (delta + math.pi / 6.0) % (math.pi / 3.0) - math.pi / 6.0
    
    # Distance is (sqrt(3)/2) / cos(delta)
    return (math.sqrt(3.0) / 2.0) / math.cos(delta)

def _smooth_penalty(inner_params, beta=10.0):
    """
    Smooth penalty for overlaps between inner hexagons.
    Uses support function distance approximation.
    """
    n = inner_params.shape[0]
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, th1 = inner_params[i]
            cx2, cy2, th2 = inner_params[j]
            
            dx = cx2 - cx1
            dy = cy2 - cy1
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist < 1e-8:
                # Centers are same, huge penalty
                return 1e6
                
            phi = math.atan2(dy, dx)
            
            # Support function of hex i in direction phi
            h1 = _support_function(phi, th1)
            # Support function of hex j in direction phi + pi
            h2 = _support_function(phi + math.pi, th2)
            
            # Gap = dist - h1 - h2
            # If gap < 0, they overlap (in this approximation)
            gap = dist - h1 - h2
            
            if gap < 0:
                # Smooth max of 0 and -gap
                # Use softplus or just square
                # penalty += (-gap) ** 2
                # To make it smoother, we can use a smooth approximation of max(0, x)
                # But for L-BFGS, x^2 is fine if we handle the kink.
                # Let's use a smooth approximation: 0.5 * (sqrt(gap^2 + eps) - eps) is not right for max.
                # max(0, -gap) = 0.5 * (-gap + sqrt(gap^2 + eps)) - 0.5*eps? No.
                # Let's just use (-gap)**2. It's differentiable everywhere except gap=0,
                # but gradient is 0 there, so L-BFGS should handle it.
                penalty += (-gap) ** 2
                
    return penalty

def _smooth_objective(params, outer_angle_deg, n, beta_side=50.0, beta_pen=10.0):
    inner_params = params.reshape(n, 3)
    side = _smooth_side(inner_params, outer_angle_deg, beta=beta_side)
    penalty = _smooth_penalty(inner_params, beta=beta_pen)
    # Weight the penalty heavily to enforce separation
    return side + 10.0 * penalty

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

    # Phase 1: Smooth Surrogate Optimization (L-BFGS-B)
    # This is the main driver. We use the smooth objective to navigate the space.
    if time.time() < DEADLINE - safety_margin - 20:
        try:
            init_params = best_inner.flatten()
            
            # We might want to try a few different beta values or restarts
            # But let's start with one strong L-BFGS-B run
            
            res = minimize(
                _smooth_objective, 
                init_params, 
                args=(best_angle, n, 50.0, 10.0), 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            if res.fun < 1e9:
                cand = res.x.reshape(n, 3)
                cand_side = _fast_side(cand, best_angle)
                if _is_valid(cand, cand_side, best_angle, min_sep=1e-8) and cand_side < best_side - 1e-12:
                    best_inner = cand
                    best_side = cand_side
        except Exception:
            pass

    # Phase 2: Simulated Annealing with Smooth Objective
    # If L-BFGS-B got stuck, SA might help escape.
    if time.time() < DEADLINE - safety_margin - 10:
        try:
            current = best_inner.copy()
            current_obj = _smooth_objective(current.flatten(), best_angle, n, 50.0, 10.0)
            best_sa = current.copy()
            best_sa_obj = current_obj
            
            T = 0.1
            T_min = 1e-5
            cooling = 0.99
            rng = np.random.RandomState(42)
            start = time.time()
            
            while T > T_min and (time.time() - start) < (DEADLINE - safety_margin - 10):
                n_perturb = rng.randint(1, 3)
                candidates = rng.choice(n, size=n_perturb, replace=False)
                new_params = current.copy()
                for idx in candidates:
                    new_params[idx, 0] += rng.randn() * 0.1
                    new_params[idx, 1] += rng.randn() * 0.1
                    new_params[idx, 2] += rng.randn() * 2.0
                
                new_obj = _smooth_objective(new_params.flatten(), best_angle, n, 50.0, 10.0)
                delta = new_obj - current_obj
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current = new_params
                    current_obj = new_obj
                    if current_obj < best_sa_obj - 1e-12:
                        best_sa = current.copy()
                        best_sa_obj = current_obj
                T *= cooling
            
            # Validate SA result
            cand_side = _fast_side(best_sa, best_angle)
            if _is_valid(best_sa, cand_side, best_angle, min_sep=1e-8) and cand_side < best_side - 1e-12:
                best_inner = best_sa
                best_side = cand_side
        except Exception:
            pass

    # Phase 3: High-precision angle polish
    if time.time() < DEADLINE - safety_margin:
        best_side, best_angle = _polish_angle(best_inner, best_angle, DEADLINE - safety_margin)

    # Phase 4: Final local refinement with Hard Objective (Nelder-Mead)
    # To ensure we are at the true minimum of the hard objective
    if time.time() < DEADLINE - safety_margin - 2:
        try:
            def hard_obj(params):
                p = params.reshape(n, 3)
                return _fast_side(p, best_angle) + 100.0 * _smooth_penalty(p, 10.0)
            
            init_params = best_inner.flatten()
            res = minimize(hard_obj, init_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            
            if time.time() < DEADLINE - safety_margin:
                cand = res.x.reshape(n, 3)
                cand_side = _fast_side(cand, best_angle)
                if _is_valid(cand, cand_side, best_angle, min_sep=1e-8) and cand_side < best_side - 1e-12:
                    best_inner = cand
                    best_side = cand_side
        except Exception:
            pass

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
