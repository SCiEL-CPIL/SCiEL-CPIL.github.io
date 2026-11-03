"""Optimized packing of 50 equal spheres in a unit cube using continuation + L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the common radius for a set of centers in [-0.5, 0.5]^3."""
    n = len(centers)
    if n < 2:
        return 0.0
    # Wall distances
    wall_dists = np.min(np.abs(centers), axis=1)
    wall_radius = np.min(wall_dists)
    # Pairwise distances
    # Use broadcasting for pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    pair_radius = min_pair_dist / 2.0
    return min(wall_radius, pair_radius)

def _soft_min_radius(centers, beta):
    """
    Smooth surrogate for the radius using soft-min.
    r = min(min_i wall_i, min_{i<j} d_ij/2)
    We use log-sum-exp for soft-min: soft_min(x) = -log(sum(exp(-beta*x)))/beta
    """
    n = len(centers)
    # Wall distances
    wall_dists = np.min(np.abs(centers), axis=1)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_radii = dists / 2.0
    
    # Combine all constraints: wall_dists and pair_radii
    # For soft-min, we want to maximize the minimum, so we minimize -soft_min
    # soft_min(x) = -log(sum(exp(-beta*x)))/beta
    # We want to maximize this, so we minimize -soft_min = log(sum(exp(-beta*x)))/beta
    
    all_vals = np.concatenate([wall_dists, pair_radii[np.triu_indices(n, k=1)]])
    
    # For numerical stability, subtract max before exp
    max_val = np.max(all_vals)
    shifted = -beta * (all_vals - max_val)
    # Clamp to avoid overflow
    shifted = np.clip(shifted, -700, 700)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + beta * max_val
    
    soft_min_val = -log_sum_exp / beta
    return soft_min_val

def _objective(centers_flat, beta):
    """Objective to minimize: -soft_min_radius."""
    centers = centers_flat.reshape(-1, 3)
    r = _soft_min_radius(centers, beta)
    return -r

def _initial_grid(n):
    """Generate an initial grid configuration."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _optimize_from_initial(centers_init, beta_schedule, max_time_per_opt=30.0):
    """Run L-BFGS-B optimization with continuation on beta."""
    centers = centers_init.copy()
    current_radius = _compute_radius(centers)
    
    for beta in beta_schedule:
        start_time = time.time()
        x0 = centers.flatten()
        
        # Bounds: centers must be in [-0.5, 0.5]^3
        bounds = [(-0.5, 0.5)] * (3 * N)
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            break
        
        elapsed = time.time() - start_time
        if elapsed > max_time_per_opt:
            break
        
        new_centers = result.x.reshape(-1, 3)
        new_radius = _compute_radius(new_centers)
        
        if new_radius > current_radius:
            centers = new_centers
            current_radius = new_radius
    
    return centers, current_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initial configuration
    centers_init = _initial_grid(N)
    centers, current_radius = _optimize_from_initial(
        centers_init,
        beta_schedule=[10, 50, 100, 200, 400],
        max_time_per_opt=25.0
    )
    
    # If we have time, try a few more refinements with higher beta
    while time.time() < deadline - 10.0:
        centers, current_radius = _optimize_from_initial(
            centers,
            beta_schedule=[800, 1600, 3200],
            max_time_per_opt=20.0
        )
        if time.time() > deadline - 5.0:
            break
    
    # Final check: ensure validity
    if not np.all(np.isfinite(centers)):
        centers = centers_init
    # Ensure centers are within bounds
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
