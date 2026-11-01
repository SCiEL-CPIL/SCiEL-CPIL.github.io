"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the maximum common radius for the given centers."""
    n = len(centers)
    if n < 2:
        return 0.5
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    # Centers must be in [-0.5, 0.5]^3
    # Distance to boundary is min(0.5 - |c_i|) for each coordinate
    min_boundary = np.min(0.5 - np.abs(centers))
    
    r = min(min_pair_dist / 2.0, min_boundary)
    return max(r, 0.0)


def _objective(x, n):
    """Objective function to minimize (negative of radius)."""
    centers = x.reshape(n, 3)
    # Clip to valid range to avoid invalid scores during optimization
    centers = np.clip(centers, -0.4999, 0.4999)
    r = _compute_radius(centers)
    return -r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centers = None
    best_radius = -1.0
    
    # Initial guess: simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_centers = grid[:N]
    
    # Try a few random restarts
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = init_centers.flatten()
        else:
            # Random perturbation of grid
            x0 = init_centers.copy()
            x0 += np.random.normal(0, 0.05, x0.shape)
            x0 = np.clip(x0, -0.49, 0.49).flatten()
        
        # Use L-BFGS-B for optimization
        result = minimize(
            _objective,
            x0,
            args=(N,),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        
        centers = np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
        r = _compute_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers
    
    # If no optimization was done or failed, return grid
    if best_centers is None:
        return init_centers
    
    return best_centers
# EVOLVE-BLOCK-END
