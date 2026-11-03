"""Packing 50 equal circles in a unit square [-0.5, 0.5]^2.

Strategy:
1. Build a good initial configuration via a grid + perturbation + local optimization.
2. Use a soft-min surrogate for the radius objective to get a smooth landscape.
3. Run scipy.optimize.minimize with L-BFGS-B on the smooth objective.
4. Polish with a high-precision local search using exact radius evaluation.
5. Time-bounded: return best found before deadline.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 110.0  # safety margin before 120s limit


def _exact_radius(centers):
    """Compute the exact common radius for a set of centers."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1],
    ]), axis=0)
    min_wall = np.min(wall_dists)

    return min(min_pair, min_wall)


def _soft_radius(centers, beta=50.0):
    """Soft-min surrogate for the radius objective (to be maximized).
    
    We minimize the negative of the soft radius.
    """
    n = len(centers)
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Soft min of pairwise distances / 2
    pair_vals = dists / 2.0
    # Use log-sum-exp trick for soft min: softmin(x) = -logsumexp(-beta * x) / beta
    # But we want to maximize the min, so we minimize -softmin
    # Let's compute softmin of pair_vals
    min_pair_soft = -np.log(np.sum(np.exp(-beta * pair_vals))) / beta
    
    # Wall distances
    wall_dists = np.stack([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1],
    ], axis=1).flatten()
    min_wall_soft = -np.log(np.sum(np.exp(-beta * wall_dists))) / beta
    
    # Overall soft radius is soft min of both
    all_vals = np.concatenate([pair_vals, wall_dists])
    overall_soft = -np.log(np.sum(np.exp(-beta * all_vals))) / beta
    
    return -overall_soft  # We minimize this


def _make_grid_init(n_circles):
    """Create a grid-based initial configuration."""
    m = int(np.ceil(np.sqrt(n_circles)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    centers = np.array([(x, y) for y in ticks for x in ticks])[:n_circles]
    return centers


def _perturb(centers, scale=0.02, seed=42):
    """Add small random perturbation."""
    rng = np.random.RandomState(seed)
    return centers + rng.randn(*centers.shape) * scale


def _local_refine(centers, beta=100.0, max_iter=2000, tol=1e-12):
    """Refine centers using L-BFGS-B on the soft objective."""
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(-1, 2)
        return _soft_radius(c, beta)
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
    return result.x.reshape(-1, 2)


def _high_precision_polish(centers, max_iter=5000):
    """High-precision polish using exact radius with smaller steps."""
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(-1, 2)
        return -_exact_radius(c)
    
    # Use BFGS for high precision
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
    return result.x.reshape(-1, 2)


def construct():
    start_time = time.time()
    
    # Start with grid
    best_centers = _make_grid_init(N)
    best_radius = _exact_radius(best_centers)
    
    # Try multiple perturbations and refinements
    seeds = [42, 123, 456, 789, 1011, 2020, 3030, 4040]
    scales = [0.01, 0.02, 0.03, 0.015, 0.025]
    
    for seed in seeds:
        if time.time() - start_time > DEADLINE - 30:
            break
        for scale in scales:
            if time.time() - start_time > DEADLINE - 30:
                break
            perturbed = _perturb(best_centers, scale, seed)
            # Ensure within bounds
            perturbed[:, 0] = np.clip(perturbed[:, 0], -0.45, 0.45)
            perturbed[:, 1] = np.clip(perturbed[:, 1], -0.45, 0.45)
            
            refined = _local_refine(perturbed, beta=50.0, max_iter=1000)
            r = _exact_radius(refined)
            if r > best_radius:
                best_radius = r
                best_centers = refined
    
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 10:
        polished = _high_precision_polish(best_centers, max_iter=3000)
        r = _exact_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished
    
    return best_centers
