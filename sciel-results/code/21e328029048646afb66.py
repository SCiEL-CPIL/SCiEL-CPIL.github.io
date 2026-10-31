"""Improved program: Direct optimization of 16 point coordinates using Differential Evolution and local refinement."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2

DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _objective(params):
    """
    Objective function for 32 parameters (16 points in 2D).
    1. Reshape to (16, 2)
    2. Center and normalize scale to remove translation/scaling invariance.
    3. Compute (max_dist / min_dist)^2.
    """
    pts = params.reshape(N, D)
    
    # Center
    pts -= pts.mean(axis=0)
    
    # Normalize scale (RMS radius = 1)
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms < 1e-10:
        return 1e10
    pts /= rms
    
    # Compute pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    
    # Exclude self-distances
    np.fill_diagonal(sq_dists, np.inf)
    
    min_sq_d = np.min(sq_dists)
    max_sq_d = np.max(sq_dists)
    
    if min_sq_d < 1e-12:
        return 1e10
        
    ratio = max_sq_d / min_sq_d
    return ratio * ratio

def construct():
    start_time = time.time()
    
    # Initial guess: 4x4 grid, centered and scaled
    grid_coords = np.array(list(np.ndindex(4, 4)), dtype=float)
    grid_coords -= grid_coords.mean(axis=0)
    # Normalize grid to RMS radius 1 to match objective normalization
    rms_grid = np.sqrt(np.mean(np.sum(grid_coords**2, axis=1)))
    grid_coords /= rms_grid
    initial_params = grid_coords.flatten()
    
    best_score = _objective(initial_params)
    best_params = initial_params.copy()
    
    # Bounds for DE: [-2, 2] for each coordinate
    bounds = [(-2.0, 2.0)] * (N * D)
    
    # Run Differential Evolution
    try:
        def de_callback(xk, convergence=False):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                raise TimeoutError("Time budget exceeded")
                
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            update_interval=1,
            callback=de_callback,
            init='sobol'
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Local Refinement
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            res = minimize(
                _objective,
                best_params,
                method='Powell',
                options={'maxiter': 200, 'maxfev': 2000},
                tol=1e-10
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # One more polish with Nelder-Mead if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN / 2:
        try:
            res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10},
                tol=1e-10
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Final point extraction
    pts = best_params.reshape(N, D)
    pts -= pts.mean(axis=0)
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms > 1e-10:
        pts /= rms
    else:
        pts = initial_params.reshape(N, D)
        
    return pts
