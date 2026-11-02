"""Optimized packing of 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def compute_radius(x):
    """Compute the actual radius for a given configuration."""
    x = x.reshape(N, 2)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    boundary = np.minimum(
        np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
        np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
    )
    
    min_pair = np.min(dist)
    min_boundary = np.min(boundary)
    
    return min(min_pair / 2.0, min_boundary)


def objective(x, sigma=0.002):
    """Smooth objective that approximates -min_distance."""
    x = x.reshape(N, 2)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    boundary = np.minimum(
        np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
        np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
    )
    
    # Get finite pairwise distances
    flat_dist = dist.flatten()
    flat_dist = flat_dist[~np.isinf(flat_dist)]
    
    # Combine pairwise and boundary distances
    all_dists = np.concatenate([flat_dist, boundary])
    
    # Soft min using log-sum-exp trick
    max_d = np.max(all_dists)
    exp_terms = np.exp(-(all_dists - max_d) / sigma)
    soft_min = max_d - sigma * np.log(np.sum(exp_terms))
    
    return -soft_min


def optimize_from_initial(x0, bounds, max_iter=1000):
    """Run L-BFGS-B optimization from a given initial point."""
    result = minimize(objective, x0.flatten(), method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
    return result.x.reshape(N, 2)


def construct():
    """Construct optimal packing of 200 circles in unit square."""
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Try multiple grid configurations and pick the best
    best_x = None
    best_r = -np.inf
    
    # Try different grid sizes and arrangements
    for m in range(14, 17):
        # Create grid points
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        
        # If we have more than N points, take a subset
        if len(grid) >= N:
            # Try different subsets: first N, or randomly sampled
            x0 = grid[:N].copy()
            
            # Add small random perturbation to break symmetry
            np.random.seed(42)
            x0 += np.random.randn(N, 2) * 0.001
            x0 = np.clip(x0, -0.49, 0.49)
            
            # Optimize
            x_opt = optimize_from_initial(x0, bounds, max_iter=800)
            x_opt = np.clip(x_opt, -0.4999, 0.4999)
            
            r = compute_radius(x_opt)
            if r > best_r:
                best_r = r
                best_x = x_opt.copy()
    
    # Also try a denser grid with different arrangement
    m = 15
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # Try shuffled grid
    np.random.seed(123)
    indices = np.random.permutation(len(grid))[:N]
    x0 = grid[indices].copy()
    x0 += np.random.randn(N, 2) * 0.002
    x0 = np.clip(x0, -0.49, 0.49)
    
    x_opt = optimize_from_initial(x0, bounds, max_iter=1000)
    x_opt = np.clip(x_opt, -0.4999, 0.4999)
    
    r = compute_radius(x_opt)
    if r > best_r:
        best_r = r
        best_x = x_opt.copy()
    
    return best_x
# EVOLVE-BLOCK-END
