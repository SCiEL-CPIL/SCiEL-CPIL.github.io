"""Program for packing 200 equal circles in a unit square using force-directed relaxation."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Initialize with a jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break grid symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, size=centers.shape)
    
    # Clamp to valid range
    centers = np.clip(centers, -0.499, 0.499)
    
    # Smooth objective: Sum of inverse squared distances
    # This is differentiable and drives the system to a local packing maximum
    def smooth_objective(x):
        c = x.reshape(-1, 2)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        # Avoid division by zero
        np.fill_diagonal(dists_sq, 1.0) 
        # Inverse squared distances
        # Add a small epsilon to prevent infinity if they get too close
        inv_dists_sq = 1.0 / (dists_sq + 1e-6)
        # Sum over all pairs (upper triangle to avoid double counting, or just sum all and divide by 2)
        # Summing all is fine, constant factor doesn't matter for optimization
        pair_energy = np.sum(inv_dists_sq)
        
        # Boundary energy: distance to walls
        # We want to maximize distance to boundary, so minimize 1/d^2
        # Distance to left/right walls: 0.5 - |x|
        # Distance to top/bottom walls: 0.5 - |y|
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        # Distance to nearest wall for each circle
        wall_dist = np.minimum(dx, dy)
        # Prevent division by zero
        wall_dist_safe = np.maximum(wall_dist, 1e-6)
        boundary_energy = np.sum(1.0 / (wall_dist_safe ** 2))
        
        return pair_energy + boundary_energy

    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Use L-BFGS-B on the smooth objective
    result = minimize(smooth_objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if result.success or result.fun < smooth_objective(x0):
        centers = result.x.reshape(-1, 2)
    
    # Final clamp to ensure validity
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
