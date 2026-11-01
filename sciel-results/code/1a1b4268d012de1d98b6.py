import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    return np.vstack([pts1, pts2])

def _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)

    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-300:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    signs[np.abs(cross) < 1e-15] = 1.0
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def _refine_full_space(init_points, deadline, max_iter=500):
    """Refine a set of points in the full 28D space using L-BFGS-B with smooth min."""
    if time.time() > deadline - 2:
        return init_points
    
    x0 = _normalize(init_points).flatten()
    
    # Use a high beta for sharpness
    beta = 500.0
    
    def _obj_and_grad(x_flat):
        return _neg_smooth_score_and_grad(x_flat, beta)
    
    try:
        res = scipy.optimize.minimize(
            _obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        return res.x.reshape(-1, 2)
    except Exception:
        return init_points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    # 1. Explore Two-Ring Configurations
    # Splits of 14 points into two rings
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3)]
    
    for n1, n2 in splits:
        if time.time() > deadline - 10:
            break
            
        # Parameters: r2 (radius of ring 2, r1 fixed at 1.0), rot1, rot2
        # We can also vary r1, but fixing r1=1 and varying r2 is equivalent up to scale
        # since the score is scale-invariant. However, the relative size matters.
        
        def _two_ring_score(params):
            r2 = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            if r2 < 0.1 or r2 > 2.0:
                return 1e6 # Penalty for invalid radii
            
            pts = _generate_two_ring_config(n1, n2, 1.0, r2, rot1, rot2)
            sc = _score(_normalize(pts))
            return -sc

        # Multi-start for the parameter optimization
        best_two_ring_points = None
        best_two_ring_score = 0.0
        
        for r_init in [0.5, 0.7, 0.9, 1.1, 1.3]:
            for rot_init in [0.0, np.pi/n1, np.pi/n2]:
                if time.time() > deadline - 10:
                    break
                x0 = np.array([r_init, rot_init, rot_init + np.pi/n2])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    sc = -res.fun
                    if sc > best_two_ring_score:
                        best_two_ring_score = sc
                        r_opt, rot1_opt, rot2_opt = res.x
                        best_two_ring_points = _generate_two_ring_config(n1, n2, 1.0, r_opt, rot1_opt, rot2_opt)
                except Exception:
                    pass
        
        if best_two_ring_points is not None:
            _update_best(best_two_ring_points)
            # Refine in full space
            if time.time() < deadline - 5:
                refined = _refine_full_space(best_two_ring_points, deadline)
                _update_best(refined)

    # 2. Explore Three-Ring Configurations (if time permits)
    three_ring_splits = [(4, 5, 5), (5, 4, 5), (5, 5, 4), (3, 5, 6), (6, 5, 3)]
    
    for n1, n2, n3 in three_ring_splits:
        if time.time() > deadline - 15:
            break
            
        def _three_ring_score(params):
            r2 = params[0]
            r3 = params[1]
            rot1 = params[2]
            rot2 = params[3]
            rot3 = params[4]
            
            if r2 < 0.1 or r3 < 0.1 or r2 > 2.0 or r3 > 2.0:
                return 1e6
                
            pts = _generate_three_ring_config(n1, n2, n3, 1.0, r2, r3, rot1, rot2, rot3)
            sc = _score(_normalize(pts))
            return -sc

        best_three_ring_points = None
        best_three_ring_score = 0.0
        
        for r2_init in [0.6, 0.8]:
            for r3_init in [1.2, 1.4]:
                if time.time() > deadline - 15:
                    break
                x0 = np.array([r2_init, r3_init, 0.0, np.pi/n2, 2*np.pi/n3])
                try:
                    res = scipy.optimize.minimize(
                        _three_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5}
                    )
                    sc = -res.fun
                    if sc > best_three_ring_score:
                        best_three_ring_score = sc
                        r2_opt, r3_opt, rot1_opt, rot2_opt, rot3_opt = res.x
                        best_three_ring_points = _generate_three_ring_config(n1, n2, n3, 1.0, r2_opt, r3_opt, rot1_opt, rot2_opt, rot3_opt)
                except Exception:
                    pass
        
        if best_three_ring_points is not None:
            _update_best(best_three_ring_points)
            if time.time() < deadline - 5:
                refined = _refine_full_space(best_three_ring_points, deadline)
                _update_best(refined)

    # 3. Fallback: Regular Polygon and Random
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
        best_score = _score(best_points)
    else:
        # Try one random start full-space refinement if time is left
        if time.time() < deadline - 10:
            r = np.sqrt(rng.uniform(0.1, 1.0, N))
            theta = rng.uniform(0, 2 * np.pi, N)
            init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            refined = _refine_full_space(init, deadline, max_iter=200)
            _update_best(refined)

    # Final normalization and check
    best_points = _normalize(best_points)
    final_score = _score(best_points)
    
    if final_score < 0.01:
        best_points = _normalize(_regular_polygon(N, 1.0))
        
    return best_points
