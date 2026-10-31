import numpy as np
from scipy.optimize import differential_evolution
import itertools

N = 16
D = 2

def pairwise_distances_sq(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def objective(x):
    points = x.reshape(-1, D)
    d_sq = pairwise_distances_sq(points)
    mask = ~np.eye(N, dtype=bool)
    d_sq_vals = d_sq[mask]
    
    if np.any(d_sq_vals == 0):
        return 1e10
    
    d_min_sq = np.min(d_sq_vals)
    d_max_sq = np.max(d_sq_vals)
    
    return d_max_sq / d_min_sq

def construct():
    # Use complex polynomial approach: P(z) = prod (z - exp(i*theta_k))
    # Roots of P(z) give us 16 points on unit circle initially
    # But we can scale and translate to optimize
    
    # Strategy: Parameterize 16 points by 16 angles theta_k
    # Points are at exp(i*theta_k) * r_k + center
    # Actually, let's use the polynomial root approach more directly
    
    # Better approach: Start with roots of unity (16th roots) and perturb
    # Then optimize positions directly using differential evolution
    
    # Initial guess: 16th roots of unity
    angles_init = np.linspace(0, 2*np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    
    # Also try hexagonal lattice subset
    hex_candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist <= 2:
                x = 1.5 * q
                y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
                hex_candidates.append((x, y))
    hex_arr = np.array(hex_candidates)
    d_orig = np.sum(hex_arr**2, axis=1)
    sorted_indices = np.argsort(d_orig)
    hex_seed = hex_arr[sorted_indices[:N]]
    hex_seed -= np.mean(hex_seed, axis=0)
    # Normalize to similar scale
    hex_seed *= 1.0 / np.max(np.linalg.norm(hex_seed, axis=1))
    
    # 4x4 grid
    grid = list(itertools.product(range(4), repeat=2))[:N]
    grid_seed = np.array(grid, dtype=float)
    grid_seed -= np.mean(grid_seed, axis=0)
    grid_seed *= 1.0 / np.max(np.linalg.norm(grid_seed, axis=1))
    
    # Define bounds for differential evolution
    # Each point has x, y coordinates
    # Use reasonable bounds
    bounds = [(-2, 2)] * (N * D)
    
    # Run differential evolution from multiple starting points
    best_score = np.inf
    best_points = None
    
    # Try from roots of unity
    try:
        result = differential_evolution(
            objective, 
            bounds,
            x0=points_init.flatten(),
            seed=42,
            maxiter=100,
            tol=1e-6,
            polish=True
        )
        if result.fun < best_score:
            best_score = result.fun
            best_points = result.x.reshape(-1, D)
    except Exception:
        pass
    
    # Try from hexagonal seed
    try:
        result = differential_evolution(
            objective, 
            bounds,
            x0=hex_seed.flatten(),
            seed=43,
            maxiter=100,
            tol=1e-6,
            polish=True
        )
        if result.fun < best_score:
            best_score = result.fun
            best_points = result.x.reshape(-1, D)
    except Exception:
        pass
    
    # Try from grid seed
    try:
        result = differential_evolution(
            objective, 
            bounds,
            x0=grid_seed.flatten(),
            seed=44,
            maxiter=100,
            tol=1e-6,
            polish=True
        )
        if result.fun < best_score:
            best_score = result.fun
            best_points = result.x.reshape(-1, D)
    except Exception:
        pass
    
    # If no optimization succeeded, return the best initial seed
    if best_points is None:
        seeds = [points_init, hex_seed, grid_seed]
        for seed in seeds:
            s = objective(seed.flatten())
            if s < best_score:
                best_score = s
                best_points = seed
    
    return best_points
