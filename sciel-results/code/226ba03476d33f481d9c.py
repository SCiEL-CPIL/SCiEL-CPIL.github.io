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
    
    smooth_min = max_area - np.log(sum_exp) / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    w = -0.5 * weights * signs
    
    d_cross_di_x = v1[:, 1] - v2[:, 1]
    d_cross_di_y = v1[:, 0] - v2[:, 0]
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    grad_points = np.zeros_like(points)
    
    dS_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dS_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dS_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dS_di)
    np.add.at(grad_points, _J, dS_dj)
    np.add.at(grad_points, _K, dS_dk)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def _generate_ring_config(radii, angles):
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _two_ring_7_7(r, delta):
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(delta, delta + 2 * np.pi, 7, endpoint=False)
    all_angles = np.concatenate([angles_outer, angles_inner])
    all_radii = np.concatenate([np.ones(7), r * np.ones(7)])
    return _generate_ring_config(all_radii, all_angles)

def _three_ring(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    a1 = np.linspace(rot1, rot1 + 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(rot2, rot2 + 2 * np.pi, n2, endpoint=False)
    a3 = np.linspace(rot3, rot3 + 2 * np.pi, n3, endpoint=False)
    all_angles = np.concatenate([a1, a2, a3])
    all_radii = np.concatenate([r1 * np.ones(n1), r2 * np.ones(n2), r3 * np.ones(n3)])
    return _generate_ring_config(all_radii, all_angles)

def _neg_two_ring(params):
    r, delta = params
    if r <= 0:
        r = 1e-6
    pts = _two_ring_7_7(r, delta)
    return -_score(_normalize(pts))

def _neg_three_ring(n1, n2, n3, params):
    r1, r2, r3, rot1, rot2, rot3 = params
    r1, r2, r3 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6)
    pts = _three_ring(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3)
    return -_score(_normalize(pts))

def _neg_four_ring(params):
    # params: r1, r2, r3, r4, rot1, rot2, rot3, rot4
    r1, r2, r3, r4, rot1, rot2, rot3, rot4 = params
    r1, r2, r3, r4 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6), max(r4, 1e-6)
    # Split: 4, 4, 3, 3
    n1, n2, n3, n4 = 4, 4, 3, 3
    a1 = np.linspace(rot1, rot1 + 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(rot2, rot2 + 2 * np.pi, n2, endpoint=False)
    a3 = np.linspace(rot3, rot3 + 2 * np.pi, n3, endpoint=False)
    a4 = np.linspace(rot4, rot4 + 2 * np.pi, n4, endpoint=False)
    all_angles = np.concatenate([a1, a2, a3, a4])
    all_radii = np.concatenate([r1 * np.ones(n1), r2 * np.ones(n2), r3 * np.ones(n3), r4 * np.ones(n4)])
    pts = _generate_ring_config(all_radii, all_angles)
    return -_score(_normalize(pts))

def _neg_five_ring(params):
    # params: r1, r2, r3, r4, r5, rot1, rot2, rot3, rot4, rot5
    r1, r2, r3, r4, r5, rot1, rot2, rot3, rot4, rot5 = params
    r1, r2, r3, r4, r5 = max(r1, 1e-6), max(r2, 1e-6), max(r3, 1e-6), max(r4, 1e-6), max(r5, 1e-6)
    # Split: 3, 3, 3, 3, 2
    n1, n2, n3, n4, n5 = 3, 3, 3, 3, 2
    a1 = np.linspace(rot1, rot1 + 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(rot2, rot2 + 2 * np.pi, n2, endpoint=False)
    a3 = np.linspace(rot3, rot3 + 2 * np.pi, n3, endpoint=False)
    a4 = np.linspace(rot4, rot4 + 2 * np.pi, n4, endpoint=False)
    a5 = np.linspace(rot5, rot5 + 2 * np.pi, n5, endpoint=False)
    all_angles = np.concatenate([a1, a2, a3, a4, a5])
    all_radii = np.concatenate([r1 * np.ones(n1), r2 * np.ones(n2), r3 * np.ones(n3), r4 * np.ones(n4), r5 * np.ones(n5)])
    pts = _generate_ring_config(all_radii, all_angles)
    return -_score(_normalize(pts))

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin

    best_score = 0.0
    best_points = None

    def _update_best(pts, sc=None):
        nonlocal best_score, best_points
        if sc is None:
            sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()
        return sc

    def _time_left():
        return deadline - time.time()

    def _try_optimize_smooth(init_points, maxiter_total=3000):
        if _time_left() < 10:
            return
        init_norm = _normalize(init_points)
        x = init_norm.flatten().copy()
        
        n_stages = 10
        betas = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0]
        
        for beta in betas:
            if _time_left() < 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': max(50, maxiter_total // n_stages), 'gtol': 1e-12, 'ftol': 1e-15}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final high-beta polish
        if _time_left() > 3:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 10000.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'gtol': 1e-14, 'ftol': 1e-16}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
                
                # Final exact Nelder-Mead polish
                if _time_left() > 2:
                    res = scipy.optimize.minimize(
                        _neg_score,
                        x,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                    )
                    _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    # Start with regular 14-gon as baseline
    reg14 = _regular_polygon(14, 1.0, 0.0)
    sc_reg = _update_best(reg14)

    # Phase 1: Coarse grid search on two-ring 7/7
    r_vals = np.linspace(0.15, 0.95, 30)
    delta_vals = np.linspace(0, 2 * np.pi / 7, 22, endpoint=False)

    best_two_ring = None
    best_two_ring_score = 0.0

    for r in r_vals:
        for delta in delta_vals:
            pts = _two_ring_7_7(r, delta)
            sc = _update_best(pts)
            if sc > best_two_ring_score:
                best_two_ring_score = sc
                best_two_ring = [r, delta]

    if best_two_ring is not None and _time_left() > 20:
        # Refine best two-ring
        try:
            res = scipy.optimize.minimize(
                _neg_two_ring,
                best_two_ring,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            pts_opt = _two_ring_7_7(res.x[0], res.x[1])
            _update_best(pts_opt)
        except Exception:
            pass

    if _time_left() < 30:
        return best_points if best_points is not None else _normalize(reg14)

    # Phase 2: Explore 3-ring and 4-ring configurations with DE
    three_ring_splits = [(7, 5, 2), (6, 6, 2), (5, 5, 4), (4, 5, 5), (7, 4, 3), (6, 5, 3)]
    
    for n1, n2, n3 in three_ring_splits:
        if _time_left() < 12:
            break
        
        bounds = [(0.1, 1.2), (0.1, 1.2), (0.1, 1.2),
                  (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
        
        try:
            for seed in [42, 123, 456]:
                if _time_left() < 8:
                    break
                de_res = scipy.optimize.differential_evolution(
                    lambda params: _neg_three_ring(n1, n2, n3, params),
                    bounds=bounds,
                    seed=seed,
                    maxiter=120,
                    tol=1e-8,
                    polish=True,
                    popsize=20
                )
                r1, r2, r3, rot1, rot2, rot3 = de_res.x
                pts_de = _three_ring(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3)
                _update_best(pts_de)
                
                if best_score > 52 and _time_left() > 12:
                    _try_optimize_smooth(pts_de, maxiter_total=4000)
        except Exception:
            pass

    # 4-ring search
    if _time_left() > 15:
        bounds4 = [(0.1, 1.2), (0.1, 1.2), (0.1, 1.2), (0.1, 1.2),
                   (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
        try:
            for seed in [42, 123]:
                if _time_left() < 8:
                    break
                de_res = scipy.optimize.differential_evolution(
                    _neg_four_ring,
                    bounds=bounds4,
                    seed=seed,
                    maxiter=100,
                    tol=1e-8,
                    polish=True,
                    popsize=18
                )
                r1, r2, r3, r4, rot1, rot2, rot3, rot4 = de_res.x
                n1, n2, n3, n4 = 4, 4, 3, 3
                a1 = np.linspace(rot1, rot1 + 2 * np.pi, n1, endpoint=False)
                a2 = np.linspace(rot2, rot2 + 2 * np.pi, n2, endpoint=False)
                a3 = np.linspace(rot3, rot3 + 2 * np.pi, n3, endpoint=False)
                a4 = np.linspace(rot4, rot4 + 2 * np.pi, n4, endpoint=False)
                all_angles = np.concatenate([a1, a2, a3, a4])
                all_radii = np.concatenate([r1 * np.ones(n1), r2 * np.ones(n2), r3 * np.ones(n3), r4 * np.ones(n4)])
                pts_de = _generate_ring_config(all_radii, all_angles)
                _update_best(pts_de)
                
                if best_score > 52 and _time_left() > 12:
                    _try_optimize_smooth(pts_de, maxiter_total=4000)
        except Exception:
            pass

    if _time_left() < 12:
        return best_points if best_points is not None else _normalize(reg14)

    # Phase 3: Full flexible optimization from best point found so far
    if best_points is not None:
        _try_optimize_smooth(best_points, maxiter_total=8000)

    # Phase 4: Final exact Nelder-Mead polish
    if best_points is not None and _time_left() > 3:
        try:
            res_final = scipy.optimize.minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(-1, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _normalize(reg14)
    else:
        best_points = _normalize(best_points)

    return best_points
