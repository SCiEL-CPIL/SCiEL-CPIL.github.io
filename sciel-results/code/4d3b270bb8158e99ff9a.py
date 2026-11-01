import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
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

def _two_heptagons(r1, r2, rot1, rot2):
    """Generate 14 points as two regular heptagons with given radii and rotations."""
    pts1 = _regular_polygon(7, r1, rot1)
    pts2 = _regular_polygon(7, r2, rot2)
    return np.vstack([pts1, pts2])

def _neg_two_heptagons_score(params):
    """Negative score for two-heptagon configuration."""
    r1, r2, rot1, rot2 = params
    if r1 < 0.01 or r2 < 0.01:
        return 1e6
    pts = _two_heptagons(r1, r2, rot1, rot2)
    sc = _score(_normalize(pts))
    return -sc

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

    def _try_optimize_smooth(init_points, beta_start=50.0, beta_end=500.0, maxiter=2000):
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        n_stages = 5
        betas = np.linspace(beta_start, beta_end, n_stages)
        
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
                    options={'maxiter': maxiter // n_stages, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        if time.time() < deadline - 3:
            try:
                def _obj_and_grad_high_beta(x_flat):
                    return _neg_smooth_score_and_grad(x_flat, 1000.0)
                
                res = scipy.optimize.minimize(
                    _obj_and_grad_high_beta,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
                
                if time.time() < deadline - 1:
                    def _neg_exact_score(x_flat):
                        return -_score(x_flat.reshape(-1, 2))
                    
                    res = scipy.optimize.minimize(
                        _neg_exact_score,
                        x,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    # === PRIMARY: Exhaustive search over two-heptagon parameter space ===
    # Only 4 parameters: r1, r2, rot1, rot2
    # Use differential evolution for global optimization
    
    def _de_two_heptagons(x):
        return _neg_two_heptagons_score(x)
    
    # Bounded parameter space
    bounds = [(0.1, 1.5), (0.1, 1.5), (-np.pi/7, np.pi/7), (-np.pi/7, np.pi/7)]
    
    try:
        de_res = scipy.optimize.differential_evolution(
            _de_two_heptagons,
            bounds,
            maxiter=500,
            tol=1e-10,
            popsize=20,
            seed=42,
            polish=True,
            init='sobol'
        )
        best_params_de = de_res.x
        best_pts_de = _two_heptagons(*best_params_de)
        sc_de = _update_best(best_pts_de)
    except Exception:
        best_params_de = None
    
    # Additional local refinements from multiple starting points
    if time.time() < deadline - 15:
        # Try various symmetric configurations
        symmetric_starts = [
            [1.0, 1.0, 0.0, 0.0],           # Same radius, aligned
            [1.0, 1.0, 0.0, np.pi/7],       # Same radius, offset by half-step
            [1.0, 0.5, 0.0, 0.0],           # Different radii
            [1.0, 0.5, 0.0, np.pi/7],
            [1.0, 0.3, 0.0, np.pi/14],
            [1.0, 0.7, 0.0, np.pi/7],
            [0.8, 0.8, 0.0, np.pi/7],
            [1.0, 0.4, np.pi/14, 0.0],
        ]
        
        for x0 in symmetric_starts:
            if time.time() > deadline - 10:
                break
            try:
                res = scipy.optimize.minimize(
                    _neg_two_heptagons_score,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                pts = _two_heptagons(*res.x)
                sc = _update_best(pts)
                
                # Refine with smooth optimization
                if sc > best_score * 0.95 and time.time() < deadline - 12:
                    _try_optimize_smooth(pts, maxiter=1500)
            except Exception:
                pass
    
    # === SECONDARY: General optimization from best two-heptagon result ===
    if best_points is not None and time.time() < deadline - 10:
        # Use the best two-heptagon config as starting point for full optimization
        _try_optimize_smooth(best_points, beta_start=100.0, beta_end=1000.0, maxiter=2000)
    
    # === TERTIARY: Random restarts for robustness ===
    for i in range(5):
        if time.time() > deadline - 8:
            break
        # Random perturbations of regular heptagon pairs
        r1 = rng.uniform(0.5, 1.5)
        r2 = rng.uniform(0.5, 1.5)
        rot1 = rng.uniform(-np.pi/7, np.pi/7)
        rot2 = rng.uniform(-np.pi/7, np.pi/7)
        init = _two_heptagons(r1, r2, rot1, rot2)
        
        # Add small random noise to break symmetry
        noise = rng.normal(0, 0.05, (N, 2))
        init = init + noise
        
        if time.time() < deadline - 12:
            _try_optimize_smooth(init, maxiter=1000)
    
    # === QUATERNARY: Coordinate descent polish ===
    if best_points is not None and time.time() < deadline - 5:
        pts = best_points.copy()
        for iteration in range(3):
            if time.time() > deadline - 2:
                break
            for i in range(N):
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
                        options={'maxiter': 60, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    pts[i, 0] = res.x[0]
                    pts[i, 1] = res.x[1]
                except Exception:
                    pass
        _update_best(pts)
    
    # === FINAL: High-precision polish ===
    if best_points is not None and time.time() < deadline - 3:
        x0 = best_points.flatten()
        try:
            def _obj_and_grad_final(x_flat):
                return _neg_smooth_score_and_grad(x_flat, 2000.0)
            
            res = scipy.optimize.minimize(
                _obj_and_grad_final,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            _update_best(res.x.reshape(-1, 2))
        except Exception:
            pass

    if best_points is None:
        # Fallback: regular 14-gon
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    # Final sanity check
    final_score = _score(best_points)
    if final_score < 1e-10:
        best_points = _normalize(_regular_polygon(N, 1.0))
        
    return best_points
