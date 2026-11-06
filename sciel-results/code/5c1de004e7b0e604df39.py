"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _log_score(points):
    """Compute log(Delta / n^n) for 8 points."""
    n = len(points)
    pts = np.array(points, dtype=float)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    diam = np.max(dists)
    if diam < 1e-15:
        return -1e10
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = dists[mask]
    log_delta = np.sum(np.log(ordered_dists))
    log_score = log_delta - n * np.log(n)
    return log_score

def _log_score_and_grad(points):
    """Compute log(Delta / n^n) and its gradient w.r.t. the 16 coordinates."""
    n = len(points)
    pts = np.array(points, dtype=float)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    diam = np.max(dists)
    if diam < 1e-15:
        return -1e10, np.zeros(2*n)
    
    diam_pair = np.unravel_index(np.argmax(dists), dists.shape)
    i_star, j_star = diam_pair
    
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = dists[mask]
    log_delta = np.sum(np.log(ordered_dists))
    log_score = log_delta - n * np.log(n)
    
    grad = np.zeros(2*n)
    
    # Gradient from sum_{i!=j} log(d_ij)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-15:
                continue
            grad[2*i] += (pts[i, 0] - pts[j, 0]) / (d * d)
            grad[2*i+1] += (pts[i, 1] - pts[j, 1]) / (d * d)
            grad[2*j] += (pts[j, 0] - pts[i, 0]) / (d * d)
            grad[2*j+1] += (pts[j, 1] - pts[i, 1]) / (d * d)
    
    # Gradient from -n*(n-1)*log(diam)
    coeff = -n * (n - 1)
    d_star = dists[i_star, j_star]
    if d_star > 1e-15:
        grad[2*i_star] += coeff * (pts[i_star, 0] - pts[j_star, 0]) / (d_star * d_star)
        grad[2*i_star+1] += coeff * (pts[i_star, 1] - pts[j_star, 1]) / (d_star * d_star)
        grad[2*j_star] += coeff * (pts[j_star, 0] - pts[i_star, 0]) / (d_star * d_star)
        grad[2*j_star+1] += coeff * (pts[j_star, 1] - pts[i_star, 1]) / (d_star * d_star)
    
    return log_score, grad

def _neg_log_score_and_grad(x):
    """Negative log-score for minimization."""
    pts = x.reshape(8, 2)
    ls, g = _log_score_and_grad(pts)
    return -ls, -g

def _score(points):
    """Compute Delta / n^n for 8 points."""
    ls = _log_score(points)
    return np.exp(ls)

def _neg_score(points):
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    from math import cos, sin, pi
    octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    best_points = np.array(octagon, dtype=float)
    best_log_score = _log_score(best_points)
    
    try:
        rng = np.random.RandomState(42)
        
        restart_configs = [
            ('none', {}),
            ('radial', {'scale': 0.1}),
            ('radial', {'scale': 0.2}),
            ('radial', {'scale': 0.05}),
            ('angular', {'scale': 0.05}),
            ('angular', {'scale': 0.1}),
            ('both', {'r_scale': 0.1, 'a_scale': 0.05}),
            ('both', {'r_scale': 0.2, 'a_scale': 0.1}),
            ('random', {'scale': 0.1}),
            ('random', {'scale': 0.2}),
        ]
        
        for restart_idx, (ptype, params) in enumerate(restart_configs):
            if time.time() > deadline - 10.0:
                break
            
            if ptype == 'none':
                x0 = best_points.flatten()
            elif ptype == 'radial':
                angles = 2 * pi * np.arange(8) / 8
                radii = 1.0 + params['scale'] * rng.randn(8)
                x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
            elif ptype == 'angular':
                angles = 2 * pi * np.arange(8) / 8 + params['scale'] * rng.randn(8)
                radii = 1.0
                x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
            elif ptype == 'both':
                angles = 2 * pi * np.arange(8) / 8 + params['a_scale'] * rng.randn(8)
                radii = 1.0 + params['r_scale'] * rng.randn(8)
                x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
            elif ptype == 'random':
                x0 = best_points.flatten() + params['scale'] * rng.randn(16)
            
            result = minimize(_neg_log_score_and_grad, x0, method='L-BFGS-B',
                            jac=True,
                            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10,
                                     'maxls': 50, 'maxfun': 5000})
            
            if time.time() > deadline - 10.0:
                break
            
            candidate = result.x.reshape(8, 2)
            diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 1e-8:
                continue
            
            cand_log_score = _log_score(candidate)
            if cand_log_score > best_log_score:
                best_log_score = cand_log_score
                best_points = candidate
        
        # Final high-precision polish pass
        if time.time() < deadline - 15.0:
            x_best = best_points.flatten()
            result = minimize(_neg_log_score_and_grad, x_best, method='L-BFGS-B',
                            jac=True,
                            options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-12,
                                     'maxls': 100, 'maxfun': 20000})
            
            candidate = result.x.reshape(8, 2)
            diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 1e-8:
                cand_log_score = _log_score(candidate)
                if cand_log_score > best_log_score:
                    best_log_score = cand_log_score
                    best_points = candidate
        
    except Exception:
        pass
    
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
