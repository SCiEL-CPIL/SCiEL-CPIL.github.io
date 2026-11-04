import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _fourier_points(params, n_harmonics=2):
    """
    Generate 14 points from a distorted regular 14-gon using Fourier series.
    params: array of length 2*(n_harmonics+1) = [a0, a1, b1, a2, b2, ...]
    where a0 is base radius, a_k, b_k are Fourier coefficients for harmonic k.
    """
    n = N
    base_angle = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # params: [a0, a1, b1, a2, b2, ...]
    a0 = params[0]
    
    radii = np.ones(n) * a0
    for k in range(1, n_harmonics + 1):
        a_k = params[2*k - 1]
        b_k = params[2*k]
        radii += a_k * np.cos(k * base_angle) + b_k * np.sin(k * base_angle)
    
    # Ensure radii are positive (convexity constraint)
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(base_angle)
    y = radii * np.sin(base_angle)
    
    return np.column_stack((x, y))

def _soft_min_score(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Returns negative score (for minimization).
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1.0
    except Exception:
        return 1.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    abs_cross = np.abs(cross)
    
    # Soft min via log-sum-exp for numerical stability
    max_val = np.max(abs_cross)
    exp_vals = np.exp(-beta * (abs_cross - max_val))
    soft_min_val = max_val + np.log(np.sum(exp_vals)) / beta
    
    return -0.5 * soft_min_val / hull_area

def _optimize_fourier(n_harmonics, time_limit, n_restarts=3):
    """
    Optimize Fourier parameters for the distorted regular polygon.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    n_params = 2 * (n_harmonics + 1)
    best_score = 0.0
    best_params = None
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initial parameters: base radius ~1, small perturbations
        if restart == 0:
            # Start with regular polygon
            init = np.zeros(n_params)
            init[0] = 1.0
        else:
            init = np.zeros(n_params)
            init[0] = 1.0 + rng.uniform(-0.2, 0.2)
            for k in range(1, n_harmonics + 1):
                init[2*k - 1] = rng.uniform(-0.3, 0.3)
                init[2*k] = rng.uniform(-0.3, 0.3)
        
        def objective(params):
            points = _fourier_points(params, n_harmonics)
            return _soft_min_score(points, beta=100.0)
        
        try:
            res = minimize(
                objective,
                init,
                method='L-BFGS-B',
                bounds=[(0.3, 2.0)] + [(-1.0, 1.0)] * n_harmonics * 2,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if res.x is not None:
                points = _fourier_points(res.x, n_harmonics)
                score = _calculate_score(points)
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass
    
    return best_score, best_params

def _optimize_full_points(points_init, time_limit, beta=100.0):
    """
    Optimize all 28 point coordinates using L-BFGS-B on soft-min surrogate,
    then Nelder-Mead on exact objective.
    """
    start_time = time.time()
    x0 = points_init.flatten()
    
    # Phase 1: L-BFGS-B on smooth surrogate
    best_score = _calculate_score(points_init)
    best_points = points_init.copy()
    
    try:
        res = minimize(
            lambda x: _soft_min_score(x.reshape(N, 2), beta),
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if res.x is not None:
            points_lbfgs = res.x.reshape(N, 2)
            score_lbfgs = _calculate_score(points_lbfgs)
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_points = points_lbfgs
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead on exact objective
    if time.time() - start_time < time_limit * 0.8:
        try:
            res_nm = minimize(
                lambda x: -_calculate_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_points = res_nm.x.reshape(N, 2)
        except Exception:
            pass
    
    return best_score, best_points

def _make_ring_points(n_out, n_in, r, theta):
    """Create a two-ring point configuration."""
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Fourier distorted regular polygon (low-dimensional search)
    for n_harmonics in [1, 2, 3]:
        if time.time() > deadline - 10.0:
            break
        
        t_limit = min(15.0, deadline - time.time() - 5.0)
        if t_limit < 5.0:
            break
        
        score, params = _optimize_fourier(n_harmonics, t_limit, n_restarts=3)
        if params is not None:
            points = _fourier_points(params, n_harmonics)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_points = points
    
    # Strategy 2: Ring configurations
    initial_configs = []
    for n_out, n_in in [(7, 7), (8, 6), (6, 8)]:
        for r in [0.3, 0.5, 0.7]:
            for theta in [0.0, np.pi/7, np.pi/6]:
                pts = _make_ring_points(n_out, n_in, r, theta)
                initial_configs.append(pts)
    
    # Sort by initial score
    scored_configs = []
    for pts in initial_configs:
        s = _calculate_score(pts)
        scored_configs.append((s, pts))
    scored_configs.sort(key=lambda x: -x[0])
    
    # Optimize top ring configs
    n_to_try = min(6, len(scored_configs))
    for i in range(n_to_try):
        if time.time() > deadline - 10.0:
            break
        
        s_init, pts_init = scored_configs[i]
        if s_init < 0.5 * best_overall_score and i > 2:
            continue
        
        t_limit = min(8.0, deadline - time.time() - 5.0)
        if t_limit < 3.0:
            break
        
        score, points = _optimize_full_points(pts_init, t_limit)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points
    
    # Final polish
    if best_overall_points is not None and time.time() < deadline - 15.0:
        t_remain = min(20.0, deadline - time.time() - 5.0)
        if t_remain > 5.0:
            score_final, points_final = _optimize_full_points(
                best_overall_points, t_remain, beta=200.0
            )
            if score_final > best_overall_score:
                best_overall_score = score_final
                best_overall_points = points_final
    
    # Fallback
    if best_overall_points is None:
        best_overall_points = _make_ring_points(7, 7, 0.5, 0.0)
        best_overall_score = _calculate_score(best_overall_points)
    
    return best_overall_points
