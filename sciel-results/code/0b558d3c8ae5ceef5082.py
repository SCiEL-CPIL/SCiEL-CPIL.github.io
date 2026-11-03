"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_radius(centers):
    """Compute the minimum radius for a set of centers in [-0.5, 0.5]^2."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to boundary
    min_boundary = np.min(0.5 - np.abs(centers))
    return min(min_pair_dist / 2.0, min_boundary)

def _objective(x):
    """Objective: negative of the minimum radius (to minimize)."""
    centers = x.reshape(N, 2)
    # Soft penalty for going out of bounds to help the optimizer
    centers_clipped = np.clip(centers, -0.5, 0.5)
    diff = centers_clipped[:, np.newaxis, :] - centers_clipped[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_boundary = np.min(0.5 - np.abs(centers_clipped))
    r = min(min_pair_dist / 2.0, min_boundary)
    # Penalize if any center is outside bounds
    out_of_bounds = np.sum(np.maximum(0, np.abs(centers) - 0.5))
    return -r + 100.0 * out_of_bounds

def _initial_grid():
    """Generate a grid initial guess."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].flatten()

def _initial_random():
    """Generate a random initial guess."""
    return np.random.uniform(-0.45, 0.45, N * 2)

def _initial_heuristic():
    """Generate a heuristic initial guess based on known packings."""
    # Start with a grid and add some perturbation
    x0 = _initial_grid()
    np.random.seed(42)
    x0 += np.random.normal(0, 0.01, x0.shape)
    return x0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_x = None
    best_r = 0.0
    
    # Try multiple initializations
    initializations = [
        _initial_grid,
        _initial_heuristic,
        _initial_random
    ]
    
    for init_func in initializations:
        if time.time() > deadline:
            break
        
        x0 = init_func()
        
        # SLSQP optimization
        try:
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            if result.success or result.fun < 0:
                centers = result.x.reshape(N, 2)
                # Clip to valid range
                centers = np.clip(centers, -0.5, 0.5)
                r = _compute_radius(centers)
                if r > best_r:
                    best_r = r
                    best_x = centers.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Final polish pass with tighter tolerance if we have time
    if best_x is not None and time.time() < deadline - 10.0:
        try:
            x_flat = best_x.flatten()
            result = minimize(
                _objective,
                x_flat,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-18}
            )
            centers = result.x.reshape(N, 2)
            centers = np.clip(centers, -0.5, 0.5)
            r = _compute_radius(centers)
            if r > best_r:
                best_r = r
                best_x = centers.copy()
        except Exception:
            pass
    
    if best_x is None:
        # Fallback to grid
        best_x = _initial_grid().reshape(N, 2)
    
    return best_x
# EVOLVE-BLOCK-END
