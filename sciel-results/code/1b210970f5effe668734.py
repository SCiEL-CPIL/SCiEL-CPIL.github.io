import numpy as np
import scipy.spatial
import scipy.optimize
import time

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

def _neg_score(points_flat):
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _soft_min_triangle_area_grad(points, beta=100.0):
    """
    Compute soft min triangle area and its gradient w.r.t. point coordinates.
    Returns (soft_min, grad) where grad has shape (n, 2).
    """
    n = len(points)
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_a = np.max(areas)
    if max_a < 1e-15:
        return 0.0, np.zeros((n, 2))
    
    # Weights for soft min
    weights = np.exp(-beta * (areas - max_a))
    sum_w = np.sum(weights)
    weights /= sum_w
    
    # sign of cross
    sign = np.sign(cross)
    
    # Gradient of area_i w.r.t. point coordinates
    # area_i = 0.5 * |cross_i|
    # d(area_i)/d(pi) = 0.5 * sign_i * (-v2_y + v1_y, v2_x - v1_x)
    # d(area_i)/d(pj) = 0.5 * sign_i * (v2_y, -v2_x)
    # d(area_i)/d(pk) = 0.5 * sign_i * (-v1_y, v1_x)
    
    grad = np.zeros((n, 2))
    
    # coeff[i] = 0.5 * sign[i] * weights[i]
    coeff = 0.5 * sign * weights
    
    # Vectorized accumulation using np.add.at
    # For pi (index _I):
    grad_i_0 = coeff * (-v2[:, 1] + v1[:, 1])
    grad_i_1 = coeff * (v2[:, 0] - v1[:, 0])
    
    # For pj (index _J):
    grad_j_0 = coeff * v2[:, 1]
    grad_j_1 = coeff * (-v2[:, 0])
    
    # For pk (index _K):
    grad_k_0 = coeff * (-v1[:, 1])
    grad_k_1 = coeff * v1[:, 0]
    
    np.add.at(grad, (_I, 0), grad_i_0)
    np.add.at(grad, (_I, 1), grad_i_1)
    np.add.at(grad, (_J, 0), grad_j_0)
    np.add.at(grad, (_J, 1), grad_j_1)
    np.add.at(grad, (_K, 0), grad_k_0)
    np.add.at(grad, (_K, 1), grad_k_1)
    
    # d(soft_min)/d(point) = -sum_i weights[i] * d(area_i)/d(point) = -grad
    soft_min = max_a - np.log(sum_w) / beta
    
    return soft_min, -grad

def _neg_soft_score_grad(points_flat, beta=100.0):
    """
    Negative soft score and its gradient for L-BFGS-B.
    Returns (value, gradient).
    """
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0, np.zeros_like(points_flat)
        
        smta, grad_smta = _soft_min_triangle_area_grad(points, beta)
        
        # score = smta / ha
        # We approximate d(ha)/d(point) = 0 for simplicity
        
        value = -(smta / ha)
        grad = -grad_smta / ha
        
        return value, grad
    except Exception:
        return 0.0, np.zeros_like(points_flat)

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

def _two_ring_7_7(r, delta):
    n_outer = 7
    n_inner = 7
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    points_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(delta, delta + 2 * np.pi, n_inner, endpoint=False)
    points_inner = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    
    return np.vstack([points_outer, points_inner])

def _neg_score_two_ring_7_7(params):
    r, delta = params
    if r <= 0:
        r = 1e-6
    pts = _two_ring_7_7(r, delta)
    pts_norm = _normalize(pts)
    return -_score(pts_norm)

def _three_ring(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def _neg_score_three_ring_752(params):
    try:
        r1, r2, r3 = params[0], params[1], params[2]
        rot1, rot2, rot3 = params[3], params[4], params[5]
        if r1 <= 0 or r2 <= 0 or r3 <= 0:
            r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
        pts = _three_ring(7, 5, 2, r1, r2, r3, rot1, rot2, rot3)
        pts_norm = _normalize(pts)
        return -_score(pts_norm)
    except Exception:
        return 0.0

def _neg_score_three_ring_662(params):
    try:
        r1, r2, r3 = params[0], params[1], params[2]
        rot1, rot2, rot3 = params[3], params[4], params[5]
        if r1 <= 0 or r2 <= 0 or r3 <= 0:
            r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
        pts = _three_ring(6, 6, 2, r1, r2, r3, rot1, rot2, rot3)
        pts_norm = _normalize(pts)
        return -_score(pts_norm)
    except Exception:
        return 0.0

def _regular_14_gon():
    return _regular_polygon(14, 1.0, 0.0)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()
    
    def _time_left():
        return deadline - time.time()
    
    # 1. Coarse grid search on two-ring 7/7
    r_vals = np.linspace(0.1, 0.95, 20)
    delta_vals = np.linspace(0, 2 * np.pi / 7, 12, endpoint=False)
    
    best_two_ring = None
    best_two_ring_score = 0.0
    
    for r in r_vals:
        for delta in delta_vals:
            sc = -_neg_score_two_ring_7_7([r, delta])
            if sc > best_two_ring_score:
                best_two_ring_score = sc
                best_two_ring = [r, delta]
    
    if best_two_ring is not None:
        pts_tr = _two_ring_7_7(best_two_ring[0], best_two_ring[1])
        _update_best(pts_tr, best_two_ring_score)
    
    if _time_left() < 25:
        return best_points if best_points is not None else _normalize(_regular_14_gon())
    
    # 2. Refine two-ring 7/7 with Nelder-Mead
    try:
        res = scipy.optimize.minimize(
            _neg_score_two_ring_7_7,
            best_two_ring,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        pts_opt = _two_ring_7_7(res.x[0], res.x[1])
        sc_opt = _score(_normalize(pts_opt))
        _update_best(pts_opt, sc_opt)
    except Exception:
        pass
    
    if _time_left() < 20:
        return best_points if best_points is not None else _normalize(_regular_14_gon())
    
    # 3. Three-ring 7/5/2 optimization
    bounds_752 = [(0.1, 1.5), (0.1, 1.5), (0.1, 1.5),
                  (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    
    try:
        de_res = scipy.optimize.differential_evolution(
            _neg_score_three_ring_752,
            bounds=bounds_752,
            seed=42,
            maxiter=100,
            tol=1e-10,
            polish=True,
            popsize=12
        )
        pts_de = _three_ring(7, 5, 2,
                             de_res.x[0], de_res.x[1], de_res.x[2],
                             de_res.x[3], de_res.x[4], de_res.x[5])
        sc_de = _score(_normalize(pts_de))
        _update_best(pts_de, sc_de)
    except Exception:
        pass
    
    if _time_left() < 15:
        return best_points if best_points is not None else _normalize(_regular_14_gon())
    
    # 4. Three-ring 6/6/2 optimization
    bounds_662 = [(0.1, 1.5), (0.1, 1.5), (0.1, 1.5),
                  (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    
    try:
        de_res_662 = scipy.optimize.differential_evolution(
            _neg_score_three_ring_662,
            bounds=bounds_662,
            seed=43,
            maxiter=100,
            tol=1e-10,
            polish=True,
            popsize=12
        )
        pts_de_662 = _three_ring(6, 6, 2,
                                  de_res_662.x[0], de_res_662.x[1], de_res_662.x[2],
                                  de_res_662.x[3], de_res_662.x[4], de_res_662.x[5])
        sc_de_662 = _score(_normalize(pts_de_662))
        _update_best(pts_de_662, sc_de_662)
    except Exception:
        pass
    
    if _time_left() < 10:
        return best_points if best_points is not None else _normalize(_regular_14_gon())
    
    # 5. Polish with L-BFGS-B using soft surrogate with gradient (vectorized)
    if best_points is not None:
        try:
            for beta in [100.0, 200.0, 500.0]:
                if _time_left() < 5:
                    break
                
                def _obj_with_grad(x, b=beta):
                    val, grad = _neg_soft_score_grad(x, b)
                    return val, grad
                
                res_lbfgs = scipy.optimize.minimize(
                    _obj_with_grad,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
                )
                pts_lbfgs = _normalize(res_lbfgs.x.reshape(-1, 2))
                sc_lbfgs = _score(pts_lbfgs)
                _update_best(pts_lbfgs, sc_lbfgs)
        except Exception:
            pass
    
    if _time_left() < 8:
        return best_points if best_points is not None else _normalize(_regular_14_gon())
    
    # 6. Full 28D local optimization from best point with Nelder-Mead
    if best_points is not None:
        try:
            res_full = scipy.optimize.minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_full = _normalize(res_full.x.reshape(-1, 2))
            sc_full = _score(pts_full)
            _update_best(pts_full, sc_full)
        except Exception:
            pass
    
    if _time_left() < 5:
        return best_points if best_points is not None else _normalize(_regular_14_gon())
    
    # 7. Final polish with L-BFGS-B on full 28D using soft surrogate with gradient
    if best_points is not None and _time_left() > 3:
        try:
            for beta in [200.0, 500.0, 1000.0]:
                if _time_left() < 3:
                    break
                
                def _obj_with_grad_final(x, b=beta):
                    val, grad = _neg_soft_score_grad(x, b)
                    return val, grad
                
                res_polish = scipy.optimize.minimize(
                    _obj_with_grad_final,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
                )
                pts_polish = _normalize(res_polish.x.reshape(-1, 2))
                sc_polish = _score(pts_polish)
                _update_best(pts_polish, sc_polish)
        except Exception:
            pass
    
    if best_points is None:
        best_points = _normalize(_regular_14_gon())
    else:
        best_points = _normalize(best_points)
        
    return best_points
