"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(centers):
    """Compute the packing radius for a set of centers in [-0.5, 0.5]^3."""
    # Wall distance
    wall = 0.5 - np.abs(centers).max(axis=1)
    min_wall = wall.min()
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(centers)
    min_pair = dists.min() / 2.0
    
    return min(min_wall, min_pair)


def _objective(centers_flat):
    """Negative score to maximize (minimize negative)."""
    centers = centers_flat.reshape(-1, 3)
    return -_score(centers)


def _penalized_objective(centers_flat, penalty=1e4):
    """Objective with soft constraints to keep centers in the cube."""
    centers = centers_flat.reshape(-1, 3)
    # Penalty for going outside [-0.5, 0.5]
    out = np.maximum(0, np.abs(centers) - 0.5)
    pen = penalty * np.sum(out**2)
    return _objective(centers_flat) + pen


def _local_optimize(centers, max_iter=200, tol=1e-10):
    """Run a local optimization to improve the packing."""
    # Use L-BFGS-B with bounds
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        _penalized_objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
    )
    
    # Project back to feasible region
    centers_opt = result.x.reshape(-1, 3)
    centers_opt = np.clip(centers_opt, -0.5, 0.5)
    
    return centers_opt


def _generate_initial_grid():
    """Generate a reasonable initial grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _perturb(centers, sigma):
    """Add random perturbation and clip to bounds."""
    noise = np.random.normal(0, sigma, size=centers.shape)
    perturbed = centers + noise
    return np.clip(perturbed, -0.49, 0.49)


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Start with grid
    best_centers = _generate_initial_grid()
    best_score = _score(best_centers)
    
    # Multiple restarts with different perturbations
    rng = np.random.default_rng(42)
    
    # First, try to optimize the grid directly
    result = _local_optimize(best_centers, max_iter=100, tol=1e-12)
    s = _score(result)
    if s > best_score:
        best_score = s
        best_centers = result
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_centers
    
    # Try several random restarts with varying perturbation sizes
    sigmas = [0.01, 0.02, 0.05, 0.08, 0.12]
    for sigma in sigmas:
        if time.time() > deadline - 10:
            break
        
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            perturbed = _perturb(best_centers, sigma)
            # Also try purely random starts occasionally
            if restart == 2 and sigma < 0.05:
                perturbed = rng.uniform(-0.45, 0.45, size=(N, 3))
            
            result = _local_optimize(perturbed, max_iter=150, tol=1e-12)
            s = _score(result)
            if s > best_score:
                best_score = s
                best_centers = result
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        result = _local_optimize(best_centers, max_iter=500, tol=1e-14)
        s = _score(result)
        if s > best_score:
            best_score = s
            best_centers = result
    
    # Ensure final result is valid (within bounds)
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
