"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    dists = cdist_fast(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)

def cdist_fast(a, b):
    """Fast cdist for small arrays."""
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

def _objective(x):
    """Objective function to minimize (negative radius)."""
    centers = x.reshape(-1, 3)
    # Check if centers are within bounds (with small tolerance)
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 1e10  # Penalty for out-of-bounds
    
    r = _compute_radius(centers)
    return -r

def _grid_init(perturb=0.0, seed=42):
    """Create a 5x5x2 grid initialization with optional perturbation."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.array([-0.25, 0.25])
    
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    if perturb > 0:
        rng = np.random.default_rng(seed)
        centers = centers + rng.normal(0, perturb, centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = None
    best_score = -np.inf
    
    # Try multiple initializations with L-BFGS-B (much faster than Nelder-Mead)
    initializations = [
        ("grid_0", _grid_init(perturb=0.0, seed=42)),
        ("grid_perturb_0.01", _grid_init(perturb=0.01, seed=42)),
        ("grid_perturb_0.02", _grid_init(perturb=0.02, seed=43)),
        ("grid_perturb_0.005", _grid_init(perturb=0.005, seed=44)),
    ]
    
    for name, centers in initializations:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        x0 = centers.flatten()
        current_score = _compute_radius(centers)
        
        if best_x is None or current_score > best_score:
            best_x = x0.copy()
            best_score = current_score
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                _objective, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if result.fun < -np.inf and np.all(np.isfinite(result.x)):
                r_opt = -result.fun
                if r_opt > best_score:
                    best_score = r_opt
                    best_x = result.x.copy()
                    
        except Exception as e:
            pass
    
    # If we have time, do a final refinement on the best solution
    if best_x is not None and time.time() - start_time < deadline - 5:
        try:
            result = minimize(
                _objective, best_x, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 100}
            )
            if result.fun < -np.inf and np.all(np.isfinite(result.x)):
                r_opt = -result.fun
                if r_opt > best_score:
                    best_score = r_opt
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    if best_x is not None:
        centers = best_x.reshape(-1, 3)
        # Ensure all points are strictly inside the cube
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Final check for validity
        if not np.all(np.isfinite(centers)):
            # Fallback to clean grid
            centers = _grid_init(perturb=0.0, seed=42)
        
        return centers
    else:
        # Fallback to clean grid
        return _grid_init(perturb=0.0, seed=42)

# EVOLVE-BLOCK-END
