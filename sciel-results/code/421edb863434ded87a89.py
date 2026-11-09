"""Circle packing: 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 200
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _time_left():
    """Return remaining time in seconds."""
    return TIME_LIMIT - (time.time() - _start_time)


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    # Distance between all pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls: square is [-0.5, 0.5]^2
    x = centers[:, 0]
    y = centers[:, 1]
    min_x_dist = np.min(np.minimum(0.5 - x, x + 0.5))
    min_y_dist = np.min(np.minimum(0.5 - y, y + 0.5))
    min_wall_dist = min(min_x_dist, min_y_dist)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return max(r, 0.0)


def _neg_radius(centers):
    """Negative radius (to be minimized)."""
    return -_compute_radius(centers)


def _smooth_neg_radius(centers, beta=50.0):
    """Smooth surrogate for negative radius using soft-min."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Soft min of pairwise distances (squared)
    # Use log-sum-exp for numerical stability
    min_sq = np.min(dists_sq)
    # Soft min: -log(sum(exp(-d^2 / beta))) / beta  approximates min
    # Actually, let's use a simpler approach:
    # soft_min(a) ≈ -1/beta * log(sum(exp(-a_i/beta)))
    exp_terms = np.exp(-dists_sq / beta)
    exp_terms[np.isinf(dists_sq)] = 0.0
    soft_min_sq = -beta * np.log(np.sum(exp_terms) + 1e-30)
    
    # Wall distances
    x = centers[:, 0]
    y = centers[:, 1]
    wall_dists = np.concatenate([0.5 - x, x + 0.5, 0.5 - y, y + 0.5])
    exp_wall = np.exp(-wall_dists / beta)
    soft_min_wall = -beta * np.log(np.sum(exp_wall) + 1e-30)
    
    # The radius is min(soft_min_sq/2, soft_min_wall)
    # Use another soft-min to combine
    soft_min_combined = -beta * np.log(np.exp(-soft_min_sq / (2*beta)) + np.exp(-soft_min_wall / beta))
    
    return soft_min_combined / beta  # This approximates the radius, we negate for minimization


def _local_optimize(centers, max_iter=500, tol=1e-8):
    """Local optimization using L-BFGS-B with smooth objective."""
    # Flatten centers
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(-1, 2)
        return _smooth_neg_radius(c, beta=100.0)
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
    return result.x.reshape(-1, 2)


def _hex_initial(n):
    """Generate hexagonal lattice initial configuration."""
    # Hexagonal packing: rows offset by half
    # For n points, use approximately sqrt(n) rows
    n_rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    centers = []
    for i in range(n_rows):
        n_cols = int(np.ceil(np.sqrt(n) * (1 + 0.5 * (i % 2))))
        for j in range(n_cols):
            if len(centers) >= n:
                break
            x = (j + 0.5 * (i % 2)) / n_cols - 0.5
            y = (i + 0.5) / n_rows - 0.5
            centers.append([x, y])
        if len(centers) >= n:
            break
    return np.array(centers[:n])


def _grid_initial(n):
    """Generate square grid initial configuration."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _random_initial(n, seed=42):
    """Generate random initial configuration inside the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 2))


def construct():
    global _start_time
    _start_time = time.time()
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initial configurations
    inits = [
        ("grid", _grid_initial(N)),
        ("hex", _hex_initial(N)),
        ("random1", _random_initial(N, seed=42)),
        ("random2", _random_initial(N, seed=123)),
        ("random3", _random_initial(N, seed=999)),
    ]
    
    for name, init_centers in inits:
        if _time_left() < 15.0:
            break
        
        # Evaluate initial
        r_init = _compute_radius(init_centers)
        
        # Local optimization with smooth surrogate
        try:
            opt_centers = _local_optimize(init_centers, max_iter=300)
            r_opt = _compute_radius(opt_centers)
            
            if r_opt > r_init:
                best_centers = opt_centers
                best_radius = r_opt
            elif best_radius < r_init:
                best_centers = init_centers
                best_radius = r_init
        except Exception:
            if r_init > best_radius:
                best_centers = init_centers
                best_radius = r_init
        
        if _time_left() < 10.0:
            break
    
    # If no good result, fall back to grid
    if best_centers is None:
        best_centers = _grid_initial(N)
    
    # Final polish: try a few more optimization steps with tighter tolerance
    if _time_left() > 5.0 and best_radius > 0.03:
        try:
            polish_centers = _local_optimize(best_centers, max_iter=200, tol=1e-10)
            r_polish = _compute_radius(polish_centers)
            if r_polish > best_radius:
                best_centers = polish_centers
                best_radius = r_polish
        except Exception:
            pass
    
    # Ensure all centers are strictly inside the square
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers


# Global start time
_start_time = time.time()
