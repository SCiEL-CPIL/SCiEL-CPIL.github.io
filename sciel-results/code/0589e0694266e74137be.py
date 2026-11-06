"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Construct centers of N=100 equal non-overlapping spheres in [-0.5, 0.5]^3
    maximizing the minimum radius. 
    Uses a smooth surrogate objective function based on the p-norm of inverse distances
    to allow gradient-based optimization (L-BFGS-B) to effectively maximize the 
    minimum pairwise and boundary distance.
    """
    n = N
    
    def compute_radius(centers):
        """Compute the actual minimum radius for scoring."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        x = centers[:, 0]
        y = centers[:, 1]
        z = centers[:, 2]
        boundary_dists = np.min(np.stack([
            0.5 - np.abs(x),
            0.5 - np.abs(y),
            0.5 - np.abs(z)
        ], axis=1), axis=1)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)

    def objective(params, p_power=40):
        """
        Objective to minimize: p-norm of inverse distances.
        Minimizing this is equivalent to maximizing the minimum distance for large p.
        """
        centers = params.reshape(n, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero, value doesn't matter as we take inverse power
        dists = np.sqrt(dists_sq)
        
        # Boundary distances
        x = centers[:, 0]
        y = centers[:, 1]
        z = centers[:, 2]
        b_x = 0.5 - np.abs(x)
        b_y = 0.5 - np.abs(y)
        b_z = 0.5 - np.abs(z)
        
        # We want to penalize small distances. 
        # If distance is 0, inverse is inf.
        # Clip distances to a small epsilon to avoid inf/nan during optimization
        eps = 1e-6
        dists = np.clip(dists, eps, None)
        b_x = np.clip(b_x, eps, None)
        b_y = np.clip(b_y, eps, None)
        b_z = np.clip(b_z, eps, None)
        
        # Sum of inverse powers
        # Note: We only need to consider unique pairs for efficiency, but N=100 is small enough
        # However, the diagonal is 1.0, so 1/1^p = 1. We should exclude diagonal.
        # Let's create a mask for i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Boundary distances for each point (3 values per point)
        all_boundary_dists = np.concatenate([b_x, b_y, b_z])
        
        # Calculate the p-norm of the inverse distances
        # term = (sum(d^-p))^(1/p)
        # We minimize this.
        
        inv_pair = 1.0 / (pair_dists ** p_power)
        inv_bound = 1.0 / (all_boundary_dists ** p_power)
        
        sum_inv = np.sum(inv_pair) + np.sum(inv_bound)
        
        # If sum is 0 (impossible with eps), return 0
        if sum_inv == 0:
            return 0.0
            
        obj_val = sum_inv ** (1.0 / p_power)
        
        return obj_val

    # Multi-start optimization
    rng = np.random.default_rng(42)
    best_r = 0.0
    best_centers = None
    
    # Initial configurations
    starts = []
    
    # 1. Random starts
    for _ in range(10):
        starts.append(rng.uniform(-0.4, 0.4, size=(n, 3)))
        
    # 2. Grid-like starts (distorted)
    m = 5 # 5x5x5 = 125 points, take 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= n:
        grid_subset = grid[:n]
        # Add some noise
        noisy_grid = grid_subset + rng.normal(0, 0.05, size=grid_subset.shape)
        starts.append(noisy_grid)
        
    # 3. Layered starts
    # Try to place them in layers
    for layer_dist in [0.2, 0.25, 0.3]:
        centers = np.zeros((n, 3))
        # Simple random distribution but constrained to be somewhat spread
        centers = rng.uniform(-0.5 + layer_dist, 0.5 - layer_dist, size=(n, 3))
        starts.append(centers)

    # Optimization settings
    bounds = [(-0.5, 0.5)] * (n * 3)
    
    for initial in starts:
        # Clip initial to bounds
        initial = np.clip(initial, -0.49, 0.49)
        
        # Use L-BFGS-B
        try:
            result = minimize(objective, initial.flatten(), method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-8, 'maxls': 20})
            
            if result.fun < 1e10: # Check for failure
                centers_opt = result.x.reshape(n, 3)
                # Ensure centers are strictly inside
                centers_opt = np.clip(centers_opt, -0.5, 0.5)
                
                r = compute_radius(centers_opt)
                if r > best_r:
                    best_r = r
                    best_centers = centers_opt
        except Exception:
            continue

    # If optimization failed completely, return a grid
    if best_centers is None:
        m = int(np.ceil(n ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:n]
        
    return best_centers
# EVOLVE-BLOCK-END
