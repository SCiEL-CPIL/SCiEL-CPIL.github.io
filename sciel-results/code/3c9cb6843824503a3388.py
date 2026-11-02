"""Packing 200 equal circles in a unit square via grid init + L-BFGS-B polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200
DEADLINE = 140.0  # safety margin before 150s kill

def _compute_radius(pts):
    """Compute the largest common radius for given centres."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(pts), 0.5)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall) / 2.0

def _objective(x, N):
    """Objective: negative radius (to minimize)."""
    pts = x.reshape(-1, 2)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(pts), 0.5)
    min_wall = np.min(wall_dists)
    r = min(min_pair, min_wall) / 2.0
    return -r

def construct():
    start_time = time.time()
    
    # Initial grid placement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts0 = grid[:N].copy()
    
    x0 = pts0.flatten()
    
    # Bounds: centres must be in [-0.5, 0.5]^2
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_x = x0.copy()
    best_r = _compute_radius(pts0)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_x.reshape(-1, 2)
    
    # L-BFGS-B optimization with tight tolerances
    try:
        result = minimize(
            _objective,
            x0,
            args=(N,),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 8000,
                'ftol': 1e-19,
                'gtol': 1e-17,
                'maxls': 50
            }
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            r_val = -result.fun
            if r_val > best_r:
                best_r = r_val
                best_x = result.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_x.reshape(-1, 2)
    
    # SLSQP active-set refinement with decreasing tolerance
    for tol_factor in [3.0, 1.5, 1.0]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        try:
            # Use tighter bounds based on current radius
            r_cur = _compute_radius(best_x.reshape(-1, 2))
            lo = -0.5 + r_cur
            hi = 0.5 - r_cur
            if lo >= hi:
                lo, hi = -0.5, 0.5
            
            bounds_tight = [(lo, hi)] * (2 * N)
            
            result2 = minimize(
                _objective,
                best_x,
                args=(N,),
                method='SLSQP',
                bounds=bounds_tight,
                options={
                    'maxiter': 500,
                    'ftol': 1e-15 / tol_factor,
                    'disp': False
                }
            )
            if result2.x is not None and np.all(np.isfinite(result2.x)):
                r_val2 = -result2.fun
                if r_val2 > best_r:
                    best_r = r_val2
                    best_x = result2.x.copy()
        except Exception:
            break
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_x.reshape(-1, 2)
    
    # Final high-precision polish with L-BFGS-B
    try:
        result3 = minimize(
            _objective,
            best_x,
            args=(N,),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 5000,
                'ftol': 1e-20,
                'gtol': 1e-18,
                'maxls': 100
            }
        )
        if result3.x is not None and np.all(np.isfinite(result3.x)):
            r_val3 = -result3.fun
            if r_val3 > best_r:
                best_r = r_val3
                best_x = result3.x.copy()
    except Exception:
        pass
    
    return best_x.reshape(-1, 2)
