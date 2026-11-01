"""Optimized packing of 100 equal spheres in a unit cube."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
CUBE_HALF = 0.5


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    # Distance to walls
    wall_dists = CUBE_HALF - np.abs(centers)
    min_wall = wall_dists.min()
    
    # Pairwise distances
    # For N=100, we can compute all pairwise distances
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.sqrt((diff ** 2).sum(axis=-1))
    # min_pair = np.min(dists[np.triu_indices(N, k=1)[0], np.triu_indices(N, k=1)[1]])
    
    # More memory efficient for N=100:
    min_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    
    return min(min_wall, min_pair / 2.0)


def _soft_min_radius(centers, beta=50.0):
    """
    Compute a soft surrogate for the minimum radius.
    Uses log-sum-exp for soft-min.
    Returns -soft_min_radius (for minimization).
    """
    centers = np.asarray(centers, dtype=np.float64)
    
    # Wall distances
    wall_dists = CUBE_HALF - np.abs(centers)
    min_wall = wall_dists.min()
    
    # Pairwise distances / 2
    # For soft min, we want to approximate min of all constraints
    # Collect all constraints: wall distances and pairwise half-distances
    # But that's 3*N + N*(N-1)/2 = 300 + 4950 = 5250 constraints
    
    # To be efficient, we'll compute a soft min over a subset or use a different approach
    # Let's use the actual min but smooth it
    
    # Actually, for optimization, let's use a penalty approach
    # or just use the min directly with gradient-free methods
    
    # For now, return the actual radius (nonsmooth)
    min_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    
    r = min(min_wall, min_pair / 2.0)
    return r


def _negative_radius(centers_flat):
    """Objective: minimize negative radius."""
    centers = centers_flat.reshape(-1, 3)
    r = _compute_radius(centers)
    return -r


def _build_initial_grid():
    """Build a good initial configuration."""
    # Use a 5x5x4 grid = 100 points
    # Or 4x5x5, etc.
    # Let's try to spread them out
    
    # Simple approach: 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    x = np.linspace(-CUBE_HALF + 0.05, CUBE_HALF - 0.05, nx)
    y = np.linspace(-CUBE_HALF + 0.05, CUBE_HALF - 0.05, ny)
    z = np.linspace(-CUBE_HALF + 0.05, CUBE_HALF - 0.05, nz)
    
    centers = []
    for zi in range(nz):
        for yi in range(ny):
            for xi in range(nx):
                centers.append([x[xi], y[yi], z[zi]])
    
    centers = np.array(centers[:N])
    return centers


def _optimize_from_init(centers_init, max_iter=1000, time_budget=30.0):
    """Run optimization from an initial configuration."""
    start_time = time.time()
    
    # Flatten
    x0 = centers_init.flatten()
    
    # Use Nelder-Mead for robustness
    result = minimize(
        _negative_radius,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': max_iter,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return None, result.x.reshape(-1, 3)
    
    return result, result.x.reshape(-1, 3)


def construct():
    """Construct centers for 100 equal spheres in unit cube."""
    global N
    
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Build initial grid
    centers_init = _build_initial_grid()
    current_best = centers_init.copy()
    current_best_r = _compute_radius(current_best)
    
    # Try multiple optimization runs with different initializations
    n_restarts = 3
    
    for restart in range(n_restarts):
        # Check time
        if time.time() > deadline - 30.0:
            break
        
        # For first restart, use the grid
        if restart == 0:
            init = current_best.copy()
        else:
            # Perturb the best solution
            init = current_best + np.random.randn(*current_best.shape) * 0.01
        
        # Optimize
        result, centers_opt = _optimize_from_init(init, max_iter=500, time_budget=30.0)
        
        if centers_opt is not None:
            r_opt = _compute_radius(centers_opt)
            if r_opt > current_best_r:
                current_best = centers_opt
                current_best_r = r_opt
        
        # Check time
        if time.time() > deadline - 20.0:
            break
    
    # Final polish: local search with smaller steps
    if time.time() < deadline - 10.0:
        # Try a few more iterations with tighter tolerance
        x0 = current_best.flatten()
        result = minimize(
            _negative_radius,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 200,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        centers_polished = result.x.reshape(-1, 3)
        r_polished = _compute_radius(centers_polished)
        if r_polished > current_best_r:
            current_best = centers_polished
            current_best_r = r_polished
    
    return current_best
