"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _objective(x):
    """Objective to minimize: negative of the radius."""
    c = x.reshape(N, 3)
    # Check if any point is outside the cube
    if np.any(c < -0.5 + 1e-12) or np.any(c > 0.5 - 1e-12):
        return 1e10
    r = _compute_radius(c)
    return -r

def _initial_configuration():
    """Create a better initial configuration using a body-centered cubic-like arrangement."""
    # Start with a grid but add some randomization to avoid local minima
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Take first N points
    c = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    c += np.random.normal(0, 0.01, c.shape)
    
    # Clip to stay within bounds
    c = np.clip(c, -0.49, 0.49)
    
    return c

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Get initial configuration
    c0 = _initial_configuration()
    
    # Flatten for scipy
    x0 = c0.flatten()
    
    # Try multiple random restarts if time permits
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # First, try a few different initializations
    for restart in range(3):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        if restart == 0:
            c_curr = c0
        else:
            # Different random seeds for diversity
            np.random.seed(42 + restart)
            c_curr = _initial_configuration()
        
        x_curr = c_curr.flatten()
        
        # Use Nelder-Mead for optimization (no gradient needed)
        result = minimize(
            _objective, 
            x_curr, 
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 10000,
                'xatol': 1e-8,
                'fatol': 1e-10
            }
        )
        
        c_opt = result.x.reshape(N, 3)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
    
    # Final verification
    if not np.all(np.isfinite(best_c)):
        return c0
    
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999, 0.4999)
    
    return best_c
# EVOLVE-BLOCK-END
