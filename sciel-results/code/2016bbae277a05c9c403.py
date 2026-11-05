import numpy as np
from scipy.optimize import minimize
import time

def _score(points):
    n = len(points)
    pts = np.array(points)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return -np.inf
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return np.exp(log_score)

def _neg_log_score(points_flat):
    n = 8
    pts = points_flat.reshape(n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return 1e10
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return -log_score

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Regular octagon
    angles = 2 * np.pi * np.arange(8) / 8
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    best_pts = octagon.copy()
    best_score = _score(octagon)
    
    rng = np.random.default_rng(42)
    
    # Try optimization from regular octagon first (local max check)
    try:
        result = minimize(_neg_log_score, octagon.flatten(), method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
        if result.fun < _neg_log_score(octagon.flatten()):
            candidate = result.x.reshape(8, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
    except Exception:
        pass

    # Random restarts
    for i in range(100):
        if time.time() > deadline:
            break
        
        # Random start in [-1, 1]^2
        x0 = rng.uniform(-1, 1, size=(8, 2))
        try:
            result = minimize(_neg_log_score, x0.flatten(), method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.success or result.fun < 100: # Check for reasonable result
                candidate = result.x.reshape(8, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score
        except Exception:
            pass

    # Scale to diameter 2
    pts = np.array(best_pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(8, dtype=bool)
    diameter = np.max(dists[mask])
    if diameter > 1e-12:
        pts = pts * (2.0 / diameter)
    
    return [[float(p[0]), float(p[1])] for p in pts]
