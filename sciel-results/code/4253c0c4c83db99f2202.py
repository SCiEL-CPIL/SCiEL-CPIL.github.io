import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
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
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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

    def _try_optimize_smooth(init_points, beta_start=50.0, beta_end=2000.0, maxiter=2000):
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        n_stages = 8
        betas = np.exp(np.linspace(np.log(beta_start), np.log(beta_end), n_stages))
        
        x = x0.copy()
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter // n_stages, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        if time.time() < deadline - 3:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 5000.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-16}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass

    # 1. Explore two-ring configurations (7+7 split is key for n=14)
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5)]
    
    for n1, n2 in splits:
        if time.time() > deadline - 10:
            break
        
        def _two_ring_score(params):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            r1 = 1.0
            r2 = r_ratio * r1
            
            if r2 <= 0.01 or r2 > 2.0:
                return -1e6
                
            pts = _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2)
            sc = _score(_normalize(pts))
            return -sc

        if n1 == 7 and n2 == 7:
            r_inits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            rot_inits = [0.0, np.pi/7, np.pi/14, 2*np.pi/7]
        else:
            r_inits = [0.3, 0.6, 0.9]
            rot_inits = [0.0, np.pi/n1]

        for r_init in r_inits:
            for rot_init in rot_inits:
                if time.time() > deadline - 10:
                    continue
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _update_best(pts)
                    
                    if sc > best_score * 0.9 and time.time() < deadline - 20:
                        _try_optimize_smooth(pts, maxiter=1500)
                        
                except Exception:
                    pass

    # 2. Regular polygon perturbations
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    noise_levels = [0.05, 0.1, 0.2]
    for noise_scale in noise_levels:
        if time.time() > deadline - 15:
            break
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            noise = rng.uniform(-noise_scale, noise_scale, (N, 2))
            init = _regular_polygon(N, 1.0) + noise
            _try_optimize_smooth(init, maxiter=1000)

    # 3. Final Polish: Coordinate Descent on Critical Triangles
    if best_points is not None and time.time() < deadline - 5:
        pts = best_points.copy()
        
        areas = 0.5 * np.abs((pts[_J] - pts[_I])[:, 0] * (pts[_K] - pts[_I])[:, 1] - (pts[_J] - pts[_I])[:, 1] * (pts[_K] - pts[_I])[:, 0])
        min_area = np.min(areas)
        critical_indices = np.where(areas <= min_area * 1.1)[0]
        
        involved_points = set()
        for idx in critical_indices:
            involved_points.add(_I[idx])
            involved_points.add(_J[idx])
            involved_points.add(_K[idx])
            
        if len(involved_points) > 10:
            involved_points = set(range(N))

        for _ in range(3):
            if time.time() > deadline - 2:
                break
            for i in sorted(involved_points):
                if time.time() > deadline - 2:
                    break
                def _neg_score_one_point(params):
                    new_pts = pts.copy()
                    new_pts[i, 0] = params[0]
                    new_pts[i, 1] = params[1]
                    return -_score(new_pts)
                
                x0 = np.array([pts[i, 0], pts[i, 1]])
                try:
                    res = scipy.optimize.minimize(
                        _neg_score_one_point,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                    pts[i, 0] = res.x[0]
                    pts[i, 1] = res.x[1]
                except Exception:
                    pass
        _update_best(pts)

    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
