import time
import numpy as np
from scipy.optimize import minimize

def _neg_log_score(params):
    """Negative log of the product of ordered pair distances after scaling to diameter 2."""
    points = params.reshape(8, 2)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    D = np.max(dists)
    if D < 1e-15:
        return 1e10
    scaled_dists = dists * (2.0 / D)
    mask = ~np.eye(8, dtype=bool)
    log_product = np.sum(np.log(scaled_dists[mask]))
    return -log_product

def _score(params):
    """Compute the actual score."""
    points = params.reshape(8, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    D = np.max(dists)
    if D < 1e-15:
        return 0.0
    scaled_dists = dists * (2.0 / D)
    mask = ~np.eye(8, dtype=bool)
    log_product = np.sum(np.log(scaled_dists[mask]))
    return np.exp(log_product - 8 * np.log(8))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Regular octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    best_score = _score(x0)
    best_params = x0.copy()
    
    np.random.seed(42)
    
    # Try different optimization strategies
    for r in range(30):
        if time.time() > deadline:
            break
        
        if r == 0:
            params = x0.copy()
        elif r == 1:
            # Two squares rotated
            a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
            a2 = a1 + np.pi/4
            params = np.column_stack([np.cos(np.concatenate([a1, a2])), 
                                       np.sin(np.concatenate([a1, a2]))]).flatten()
        else:
            # Random perturbation of octagon or random points
            if r % 2 == 0:
                params = x0 + np.random.randn(16) * 0.1
            else:
                params = np.random.randn(16) * 1.5
        
        try:
            result = minimize(
                _neg_log_score,
                params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if time.time() > deadline:
                break
            
            if result.fun < -np.log(best_score) - 1e-9:  # Better
                s = _score(result.x)
                if s > best_score:
                    best_score = s
                    best_params = result.x.copy()
        except Exception:
            continue
    
    best_points = best_params.reshape(8, 2)
    return [[float(x), float(y)] for x, y in best_points]
