"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import time
import sys

def _compute_radius(centers):
    """Compute the valid packing radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds: centers must be in [-0.5, 0.5]^3
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Clip to valid range for computation
    centers_clipped = np.clip(centers, -0.5, 0.5)
    
    # Wall distance: min over i,k of (0.5 - |c_ik|)
    wall_dists = 0.5 - np.abs(centers_clipped)
    min_wall = float(np.min(wall_dists))
    
    if min_wall < 0:
        return 0.0
    
    # Pairwise distances
    # Compute all pairwise distances efficiently
    diff = centers_clipped[:, np.newaxis, :] - centers_clipped[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists))
    
    # Radius is half the minimum of pairwise distances and wall distances
    r = min(min_pair / 2.0, min_wall)
    return max(0.0, r)

def _objective_and_grad(centers_flat):
    """Objective: maximize r. We minimize -r using a smooth approximation.
    For the gradient, we use subgradient of the min function."""
    centers = centers_flat.reshape(-1, 3)
    
    # Clip to valid bounds
    centers_clipped = np.clip(centers, -0.5, 0.5)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers_clipped)  # shape (N, 3)
    min_wall_idx = np.argmin(wall_dists, axis=1)  # which coordinate is limiting
    min_wall = np.min(wall_dists, axis=1)  # shape (N,)
    
    # Pairwise distances
    diff = centers_clipped[:, np.newaxis, :] - centers_clipped[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_idx_i, min_pair_idx_j = np.unravel_index(np.argmin(dists), dists.shape)
    min_pair = dists[min_pair_idx_i, min_pair_idx_j]
    
    r = min(min_pair / 2.0, min_wall)
    
    # Subgradient
    grad = np.zeros_like(centers_flat)
    
    if min_pair / 2.0 <= min_wall + 1e-15:
        # Pairwise distance is limiting
        i, j = int(min_pair_idx_i), int(min_pair_idx_j)
        if i != j and dists[i, j] > 1e-12:
            direction = (centers_clipped[i] - centers_clipped[j]) / dists[i, j]
            # d(-r)/dc_i = d(-d_ij/2)/dc_i = -0.5 * direction
            grad[3*i:3*i+3] = 0.5 * direction
            grad[3*j:3*j+3] = -0.5 * direction
    else:
        # Wall distance is limiting
        # Find all spheres at the minimum wall distance
        for k in range(N):
            if abs(wall_dists[k, min_wall_idx[k]] - min_wall) < 1e-12:
                coord = min_wall_idx[k]
                # d(-r)/dc_ik = -d(0.5 - |c_ik|)/dc_ik = sign(c_ik)
                if centers_clipped[k, coord] >= 0:
                    grad[3*k + coord] = -1.0
                else:
                    grad[3*k + coord] = 1.0
    
    return -r, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial guess: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_centers = grid[:N].copy()
    
    best_centers = init_centers.copy()
    best_r = _compute_radius(best_centers)
    
    # Try multiple restarts with perturbations
    np.random.seed(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = init_centers.flatten()
        else:
            # Perturb from best found
            x0 = best_centers.flatten() + np.random.normal(0, 0.02, size=N*3)
        
        # Define bounds
        bounds = Bounds(-0.5, 0.5)
        
        try:
            result = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -best_r:
                new_centers = np.clip(result.x.reshape(-1, 3), -0.5, 0.5)
                new_r = _compute_radius(new_centers)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centers = new_centers.copy()
                    
        except Exception:
            continue
        
        if time.time() > deadline - 20.0:
            break
    
    # Final polish: local refinement
    if time.time() < deadline - 5.0:
        try:
            x0 = best_centers.flatten()
            bounds = Bounds(-0.5, 0.5)
            result = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_centers = np.clip(result.x.reshape(-1, 3), -0.5, 0.5)
            new_r = _compute_radius(new_centers)
            if new_r > best_r + 1e-12:
                best_r = new_r
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
