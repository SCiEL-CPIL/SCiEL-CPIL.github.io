"""Optimized planar point configuration for maximum product of distances at fixed diameter."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_score(points):
    """Compute Delta/n^n for 8 points, scaled to diameter 2."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Diameter
    diam = np.max(dists)
    if diam < 1e-12:
        return 0.0
    # Scale to diameter 2
    scale = 2.0 / diam
    scaled_dists = dists * scale
    # Product of all ordered pair distances (i != j)
    # Product over ordered pairs = (product over unordered pairs) ^ 2
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    prod_ordered = 1.0
    for i in range(n):
        for j in range(i+1, n):
            prod_ordered *= scaled_dists[i, j] ** 2
    delta = prod_ordered
    score = delta / (n ** n)
    return score

def _neg_log_score_16d(x):
    """Negative log score for 16D vector."""
    pts = x.reshape(-1, 2)
    n = 8
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    dsq_upper = dists_sq[iu]
    if np.min(dsq_upper) < 1e-30:
        return 1e10
    diam_sq = np.max(dsq_upper)
    if diam_sq < 1e-24:
        return 1e10
    log_sum_dsq = np.sum(np.log(dsq_upper))
    log_diam_sq = np.log(diam_sq)
    c = n * (n - 1) // 2
    log_prod_ordered = 2.0 * c * np.log(2.0) + log_sum_dsq - c * log_diam_sq
    return n * np.log(n) - log_prod_ordered

def _neg_log_score_4d_symmetric(v):
    """Negative log score for 4D symmetric parameterization."""
    a, b, c, d = v
    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return 1e10
    
    # Construct 8 points
    # Order doesn't matter for the score, but let's keep them distinct
    pts = np.array([
        [a, b],
        [-a, b],
        [-a, -b],
        [a, -b],
        [c, d],
        [-c, d],
        [-c, -d],
        [c, -d]
    ])
    
    # Check for distinctness
    # If a==c and b==d, points 1-4 and 5-8 overlap.
    # If a==c and b!=d, points 1 and 5 are distinct, but 1 and 7?
    # (a,b) vs (c,-d) -> (a,b) vs (a,-d). Distinct if b!=d.
    # (a,b) vs (-c,d) -> (a,b) vs (-a,d). Distinct if a!=0 or b!=d.
    # Main risk: a==c and b==d.
    if np.isclose(a, c) and np.isclose(b, d):
        return 1e10
    if np.isclose(a, c) and np.isclose(b, -d): # b,d > 0 so b != -d
        pass
        
    # Calculate score
    return _neg_log_score_16d(pts.flatten())

def construct():
    deadline = time.time() + 145
    n = 8
    
    # 1. Global Search in Symmetric 4D Space
    # Bounds: coordinates must be positive. Max coord 1.0 (since diam=2, max dist 2, so max coord 1)
    bounds = [(0.05, 1.0), (0.05, 1.0), (0.05, 1.0), (0.05, 1.0)]
    
    best_4d_result = None
    best_4d_val = 1e10
    
    try:
        # Use Differential Evolution for global optimization in 4D
        # This is robust and handles the non-convexity well
        de_result = differential_evolution(
            _neg_log_score_4d_symmetric, 
            bounds=bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20,
            tol=1e-12,
            polish=True,
            init='latinhypercube'
        )
        
        if de_result.fun < best_4d_val:
            best_4d_val = de_result.fun
            best_4d_result = de_result.x
            
    except Exception:
        pass

    # 2. Multi-start Local Optimization in 4D Space (for extra safety)
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        # Generate random starts in 4D
        n_starts = 50
        for i in range(n_starts):
            if time.time() > deadline - 30:
                break
            x0 = rng.uniform(0.1, 1.0, size=4)
            try:
                res = minimize(_neg_log_score_4d_symmetric, x0, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12})
                if res.fun < best_4d_val:
                    best_4d_val = res.fun
                    best_4d_result = res.x
            except Exception:
                pass

    # 3. Convert best 4D result to 16D points
    best_points_16d = None
    if best_4d_result is not None:
        a, b, c, d = best_4d_result
        pts = np.array([
            [a, b],
            [-a, b],
            [-a, -b],
            [a, -b],
            [c, d],
            [-c, d],
            [-c, -d],
            [c, -d]
        ])
        best_points_16d = pts
    else:
        # Fallback to regular octagon
        k_vals = np.arange(n)
        base_angle = 2 * np.pi / n
        best_points_16d = np.column_stack([np.cos(base_angle * k_vals), np.sin(base_angle * k_vals)])

    # 4. Full 16D Polish
    # Allow the points to move freely to break symmetry if it improves the score
    if time.time() < deadline - 5:
        x0 = best_points_16d.flatten()
        try:
            res_16d = minimize(_neg_log_score_16d, x0, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14})
            if res_16d.fun < _neg_log_score_16d(x0):
                best_points_16d = res_16d.x.reshape(-1, 2)
        except Exception:
            pass
        
        # Nelder-Mead for final non-smooth polish
        if time.time() < deadline - 2:
            try:
                res_nm = minimize(_neg_log_score_16d, best_points_16d.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14})
                if res_nm.fun < _neg_log_score_16d(best_points_16d.flatten()):
                    best_points_16d = res_nm.x.reshape(-1, 2)
            except Exception:
                pass

    # 5. Final Verification and Formatting
    sc = _compute_score(best_points_16d)
    
    # Fallback if something went wrong
    if sc <= 0 or not np.isfinite(sc):
        k_vals = np.arange(n)
        base_angle = 2 * np.pi / n
        octagon = np.column_stack([np.cos(base_angle * k_vals), np.sin(base_angle * k_vals)])
        return [[float(octagon[i, 0]), float(octagon[i, 1])] for i in range(n)]
        
    if np.max(np.abs(best_points_16d)) >= 1e6:
        k_vals = np.arange(n)
        base_angle = 2 * np.pi / n
        octagon = np.column_stack([np.cos(base_angle * k_vals), np.sin(base_angle * k_vals)])
        return [[float(octagon[i, 0]), float(octagon[i, 1])] for i in range(n)]

    return [[float(best_points_16d[i, 0]), float(best_points_16d[i, 1])] for i in range(n)]
# EVOLVE-BLOCK-END
