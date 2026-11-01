"""Program for packing 200 equal circles in a unit square using force-directed relaxation."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Initialize with a perturbed grid to break symmetry
    m = int(np.ceil(np.sqrt(N)))
    # Use m x m grid, take first N points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break grid symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.001, size=centers.shape)
    
    # Clamp to valid range
    centers = np.clip(centers, -0.499, 0.499)
    
    # Force-directed relaxation to maximize minimum distance
    def objective(x):
        """Negative of the minimum radius (to minimize)."""
        c = x.reshape(-1, 2)
        # Compute all pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        min_boundary = np.min(np.minimum(dx, dy))
        
        # Radius is half the minimum of pair distances and boundary distances
        r = 0.5 * min(min_pair_dist, min_boundary)
        return -r
    
    # Run optimization with L-BFGS-B
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if result.success or result.fun < objective(x0):
        centers = result.x.reshape(-1, 2)
    
    # Final clamp to ensure validity
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
