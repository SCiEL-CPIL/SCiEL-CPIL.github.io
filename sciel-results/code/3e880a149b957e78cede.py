"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_radius(c):
    """Compute the maximum common radius for centres c (N,3)."""
    if c.shape[0] != N:
        return 0.0
    # Check bounds: all coords in [-0.5, 0.5]
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf so it doesn't affect min
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Distance to walls
    wall_dist = np.min(0.5 - np.abs(c), axis=1)
    min_wall = np.min(wall_dist)
    r = min(min_pair / 2.0, min_wall)
    return r

def _objective(params):
    """Minimize negative radius (i.e., maximize radius)."""
    c = params.reshape(N, 3)
    return -_compute_radius(c)

def _grid_init():
    """Create a 5x5x5 grid, take first 100 points."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_c = _grid_init()
    best_r = _compute_radius(best_c)
    
    # Multi-start optimization
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Perturb the grid initialization
        if restart == 0:
            c0 = best_c.copy()
        else:
            c0 = _grid_init()
            # Add small random perturbation
            noise_scale = 0.01 * restart
            c0 = c0 + noise_scale * np.random.randn(N, 3)
            # Clip to stay in bounds
            c0 = np.clip(c0, -0.49, 0.49)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _objective,
                c0.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (3 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        c_opt = result.x.reshape(N, 3)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt
    
    # Final polish: use the best result and try a few more iterations with tighter tolerance
    if time.time() < deadline - 5.0 and best_c is not None:
        try:
            result = minimize(
                _objective,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (3 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_polished = result.x.reshape(N, 3)
            r_polished = _compute_radius(c_polished)
            if r_polished > best_r:
                best_c = c_polished
                best_r = r_polished
        except Exception:
            pass
    
    # Ensure final result is valid
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
