import numpy as np
from scipy.optimize import minimize
import itertools
import time

_DEADLINE = time.time() + 590.0

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _soft_min_objective(params, beta=50.0):
    """
    params: 
    - 12 values for angular gaps (positive, will be normalized to sum 2*pi)
    - 2 values for interior point (x, y)
    """
    try:
        if time.time() > _DEADLINE - 10:
            return 1e6
        
        gaps = params[:12]
        # Ensure gaps are positive for normalization
        gaps = np.maximum(gaps, 1e-6)
        gaps = gaps / np.sum(gaps) * 2 * np.pi
        
        # Calculate angles
        angles = np.cumsum(np.concatenate([[-2*np.pi/12], gaps])) # Centering? No, just start at 0
        angles = np.cumsum(gaps)
        angles = np.concatenate([[0], angles[:-1]]) # Start at 0, relative angles
        
        # Hull points on unit circle
        hull_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Interior point
        ix, iy = params[12], params[13]
        
        # Combine points
        points = np.vstack([hull_pts, [[ix, iy]]])
        
        # Check if interior point is actually inside (optional, but good for sanity)
        # For now, we just compute the score. If the interior point is outside,
        # the hull will change, but the convex hull calculation handles it.
        # However, our parameterization assumes 12 hull points.
        # If the interior point moves outside, the "hull" in our parameterization
        # is not the true convex hull of the 13 points.
        # To be robust, we should check if the point is inside the 12-gon.
        # A simple check: if the point is outside, we might want to penalize it.
        # But L-BFGS-B will naturally push it in if the score is better.
        
        ha = _hull_area(points)
        if ha <= 0:
            return 1e6
            
        n = len(points)
        idx = np.array(list(itertools.combinations(range(n), 3)))
        a = points[idx[:, 0]]
        b = points[idx[:, 1]]
        c = points[idx[:, 2]]
        areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                             + b[:, 0] * (c[:, 1] - a[:, 1])
                             + c[:, 0] * (a[:, 1] - b[:, 1]))
        
        # Soft min
        # To avoid overflow/underflow in exp, we shift by max
        max_a = np.max(areas)
        # We want -soft_min / ha
        # soft_min(a) = -1/beta * log(sum(exp(-beta * a)))
        # Let's use the stable version:
        # min(a) approx = max_a - 1/beta * log(sum(exp(-beta * (a - max_a))))
        # Wait, standard log-sum-exp for min:
        # m = min(a)
        # log(sum(exp(-beta * a))) = -beta * m + log(sum(exp(-beta * (a - m))))
        # This is getting complicated. Let's use the max-based soft min for min:
        # soft_min(a) = -1/beta * log( sum( exp(-beta * a_i) ) )
        # To compute this stably:
        # Let M = max(a). 
        # sum(exp(-beta * a_i)) = exp(-beta * M) * sum(exp(-beta * (a_i - M)))
        # log(sum(...)) = -beta * M + log(sum(exp(-beta * (a_i - M))))
        # soft_min = -1/beta * ( -beta * M + log(...) ) = M - 1/beta * log(sum(exp(-beta * (a_i - M))))
        
        shifted = areas - max_a
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min_val = max_a - log_sum / beta
        
        score = -soft_min_val / ha
        return score
        
    except Exception:
        return 1e6

def _optimize_parameterized(start_params, beta=50.0):
    """
    Optimize the 14 parameters (12 gaps, 2 interior coords).
    """
    bounds = [(1e-4, 1.0)] * 12 + [(-1.0, 1.0), (-1.0, 1.0)]
    
    try:
        result = minimize(
            _soft_min_objective,
            start_params,
            args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 400, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        if result.success or result.fun < _soft_min_objective(start_params, beta):
            return result.x
    except Exception:
        pass
    return start_params

def _params_to_points(params):
    gaps = params[:12]
    gaps = np.maximum(gaps, 1e-6)
    gaps = gaps / np.sum(gaps) * 2 * np.pi
    angles = np.cumsum(gaps)
    angles = np.concatenate([[0], angles[:-1]])
    hull_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    ix, iy = params[12], params[13]
    points = np.vstack([hull_pts, [[ix, iy]]])
    return points

def _polish_raw(points0, max_iter=1000):
    """
    Final polish on raw coordinates.
    """
    pts_flat = points0.flatten()
    n = len(points0)
    bounds = [(0.0, 1.0)] * (2 * n) # Loose bounds, we will clip later if needed
    
    def _raw_obj(x):
        pts = x.reshape(-1, 2)
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        ma = _min_triangle_area(pts)
        if ma <= 0:
            return 1e6
        return -ma / ha

    try:
        result = minimize(
            _raw_obj,
            pts_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        if result.success or result.fun < _raw_obj(pts_flat):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return points0

def heilbronn_convex13() -> np.ndarray:
    n = 13
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = -np.inf
    
    def _update(pts):
        nonlocal best_points, best_score
        # Ensure finite
        if not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Baseline: Regular 12-gon + Center
    # Gaps are all 2*pi/12
    base_gaps = np.full(12, 2 * np.pi / 12)
    # Normalize to sum 2*pi (they already do)
    # Interior point at (0,0)
    base_params = np.concatenate([base_gaps, [0.0, 0.0]])
    
    # Optimization runs with different betas and starts
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > _DEADLINE - 60:
            break
            
        # Run optimization from base
        opt_params = _optimize_parameterized(base_params, beta)
        pts = _params_to_points(opt_params)
        _update(pts)
        
        # Try some random perturbations of the base params
        for _ in range(5):
            if time.time() > _DEADLINE - 30:
                break
            perturbed_gaps = base_gaps + rng.normal(0, 0.1, 12)
            perturbed_gaps = np.maximum(perturbed_gaps, 0.01)
            perturbed_int = rng.uniform(-0.2, 0.2, 2)
            start_params = np.concatenate([perturbed_gaps, perturbed_int])
            
            opt_params_p = _optimize_parameterized(start_params, beta)
            pts_p = _params_to_points(opt_params_p)
            _update(pts_p)
            
            # Polish the result
            if time.time() < _DEADLINE - 15:
                polished = _polish_raw(pts_p, max_iter=200)
                _update(polished)
                
    # If we found a good point, do a final high-precision polish on the best one
    if best_points is not None and time.time() < _DEADLINE - 10:
        final_polish = _polish_raw(best_points, max_iter=1000)
        _update(final_polish)
        
    if best_points is None:
        # Fallback to regular 12-gon + center
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        hull_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = np.vstack([hull_pts, [[0.0, 0.0]]])
        
    return best_points

# EVOLVE-BLOCK-END
