<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(positions):
    """Compute the maximum common radius for given circle centers."""
    # Boundary constraint
    min_bound = 0.5 - np.max(np.abs(positions))
    
    # Pairwise distance constraint
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(min_bound, min_dist / 2.0)
=======
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(positions):
    """Compute the maximum common radius for given circle centers."""
    # Boundary constraint
    min_bound = 0.5 - np.max(np.abs(positions))
    
    # Pairwise distance constraint - use squared distances for speed
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = np.min(dists_sq)
    
    return min(min_bound, np.sqrt(min_dist_sq) / 2.0)

def negative_radius(positions):
    """Negative of radius for minimization."""
    return -compute_radius(positions)

def polish_positions(positions, time_budget=30.0):
    """Polish positions using Nelder-Mead on the best configuration."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    current = positions.copy()
    best = positions.copy()
    best_r = compute_radius(best)
    
    # Use Nelder-Mead to maximize radius (minimize negative radius)
    result = minimize(
        negative_radius,
        current,
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }
    )
    
    if result.fun < -best_r:  # result.fun is negative radius
        new_r = -result.fun
        if new_r > best_r:
            best = result.x
            best_r = new_r
    
    # Check if we have time for a second polish pass with tighter settings
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 10:
            result2 = minimize(
                negative_radius,
                best,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if result2.fun < -best_r:
                new_r = -result2.fun
                if new_r > best_r:
                    best = result2.x
                    best_r = new_r
    
    return best
>>>>>>> REPLACE

