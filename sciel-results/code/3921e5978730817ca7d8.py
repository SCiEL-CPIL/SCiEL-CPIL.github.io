"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return float(min_dist_to_wall)
    
    # Compute pairwise distances using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Minimum distance to walls
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_energy_grad(centers, beta):
    """Compute soft repulsion energy and gradient.
    
    Energy: sum_{i<j} exp(-beta * d_ij) + sum_i exp(-beta * w_i)
    """
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Avoid division by zero in gradient
    safe_dists = np.where(dists < 1e-8, 1e-8, dists)
    
    # exp(-beta * d)
    exp_terms = np.exp(-beta * dists)
    
    # Gradient: -beta * exp(-beta * d) * (c_i - c_j) / d
    # We only need upper triangle to avoid double counting in energy, 
    # but for gradient accumulation, we can iterate over pairs.
    
    iu = np.triu_indices(n, k=1)
    dists_iu = dists[iu]
    exp_iu = exp_terms[iu]
    safe_dists_iu = safe_dists[iu]
    diff_iu = diff[iu] # shape (k, 3)
    
    # Gradient contribution for pair (i, j):
    # d/dc_i exp(-beta d) = -beta exp(-beta d) * (c_i - c_j) / d
    # d/dc_j exp(-beta d) = -beta exp(-beta d) * (c_j - c_i) / d = +beta exp(-beta d) * (c_i - c_j) / d
    
    coeff = beta * exp_iu / safe_dists_iu # shape (k,)
    
    # Accumulate gradient
    vi, vj = iu[0], iu[1]
    # Use np.add.at for efficient accumulation
    # We need to broadcast coeff to (k, 3) to multiply with diff_iu (k, 3)
    coeff_3d = coeff[:, np.newaxis] * diff_iu
    
    np.add.at(grad, vi, -coeff_3d)
    np.add.at(grad, vj, coeff_3d)
    
    energy += np.sum(exp_iu)
    
    # Wall repulsion
    wall_dists = 0.5 - np.abs(centers)
    # Avoid negative wall distances in exp? No, exp is defined for all.
    # But if center is outside, wall_dist < 0. 
    # The bounds in optimizer prevent this, but let's be safe.
    exp_wall = np.exp(-beta * wall_dists)
    energy += np.sum(exp_wall)
    
    # Gradient for wall:
    # d/dc exp(-beta w) = -beta exp(-beta w) * d/dc w
    # w = 0.5 - |c|
    # d/dc w = -sign(c)
    # So grad = beta exp(-beta w) sign(c)
    
    signs = np.sign(centers)
    # Handle c=0
    signs[centers == 0] = 1.0
    
    grad += beta * exp_wall[:, np.newaxis] * signs
    
    return energy, grad

def _optimize_soft(centers_init, beta, max_iter=200, tol=1e-12):
    """Optimize using L-BFGS-B with soft repulsion objective."""
    bounds = [(-0.5, 0.5)] * (len(centers_init) * 3)
    
    def objective_and_grad(x):
        c = x.reshape(-1, 3)
        e, g = _soft_energy_grad(c, beta)
        return e, g.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            centers_init.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'disp': False}
        )
        return result.x.reshape(-1, 3), result.success, result.fun
    except Exception:
        return centers_init, False, np.inf

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_centers = None
    best_radius = 0.0
    
    # Initial best from grid
    grid_init = _grid_init(N)
    r_grid = _compute_radius(grid_init)
    best_radius = r_grid
    best_centers = grid_init.copy()
    
    # Multi-scale optimization
    # Start with low beta (smooth) to find good global structure
    # Then increase beta to sharpen the minimum distance
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    # Try a few random starts for the first (softest) stage
    for seed in range(5):
        if time.time() > deadline - 20.0:
            break
            
        init = _random_init(N, seed=seed)
        
        # Anneal betas
        current_centers = init
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
                
            try:
                optimized, success, energy = _optimize_soft(
                    current_centers, beta, max_iter=100, tol=1e-10
                )
                current_centers = optimized
            except Exception:
                continue
            
            # Check radius
            r_opt = _compute_radius(current_centers)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = current_centers.copy()
        
        # If we found a better solution, use it as the starting point for next random seed?
        # No, we want diversity. But we can keep the best found so far as a candidate.
    
    # Final polish on the best solution found with high beta
    if best_centers is not None and time.time() < deadline - 5.0:
        # Use a very high beta for final polish
        try:
            optimized, success, energy = _optimize_soft(
                best_centers, 1000.0, max_iter=200, tol=1e-12
            )
            r_opt = _compute_radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_centers is None:
        best_centers = _grid_init(N)
    
    # Final validation: ensure all centers are strictly inside the cube
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
