import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _potential(x, N):
    """
    Calculate the repulsive potential energy.
    We use a high-power inverse distance to strongly penalize close approaches.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    # Compute squared distances to avoid sqrt in the sum if possible, 
    # but we need d for the power. 
    # d_ij^2 = sum (xi - xj)^2
    # We compute the full matrix of squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Mask the diagonal (self-distance)
    np.fill_diagonal(dist_sq, 1.0)
    
    # We want to minimize sum 1/d^p. 
    # 1/d^p = (1/d^2)^(p/2)
    # Let p=6, so we minimize sum (1/d^2)^3
    # This is equivalent to sum 1/d^6
    inv_dist_sq = 1.0 / dist_sq
    pair_energy = np.sum(inv_dist_sq**3)  # sum 1/d^6
    
    # Wall constraints: distance to walls
    # Center must be in [-0.5, 0.5]
    # Distance to wall is min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
    # Actually, the constraint is 1/2 - |c_ik| >= r.
    # To keep them inside, we penalize proximity to walls.
    # Wall distance for component k is 0.5 - |c_k|
    # We want to maximize the minimum wall distance.
    # So we minimize sum 1/(0.5 - |c_k|)^p
    # If a sphere is at the wall, this goes to infinity.
    
    # Calculate distances to each wall
    # x-coord: 0.5 - x, 0.5 + x -> min is 0.5 - |x|
    wall_dists = np.minimum(0.5 - np.abs(centers), 0.5) # This is wrong, 0.5 - |x| is the dist to the nearest wall in that axis
    # Actually, dist to wall in x is min(0.5 - x, 0.5 + x) = 0.5 - |x|
    # If x is outside [-0.5, 0.5], this becomes negative. 
    # We should clip or handle boundaries carefully.
    # L-BFGS-B has bounds. We will set bounds to [-0.5, 0.5].
    # So |x| <= 0.5.
    # dist_to_wall_x = 0.5 - np.abs(centers[:, 0])
    # dist_to_wall_y = 0.5 - np.abs(centers[:, 1])
    # dist_to_wall_z = 0.5 - np.abs(centers[:, 2])
    
    # Combine into a single vector of all wall distances
    wall_dists_vec = np.concatenate([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1]),
        0.5 - np.abs(centers[:, 2])
    ])
    
    # Avoid division by zero if a sphere is exactly on the wall
    # Add a small epsilon
    wall_dists_safe = np.maximum(wall_dists_vec, 1e-6)
    
    wall_energy = np.sum(1.0 / (wall_dists_safe**6))
    
    return pair_energy + wall_energy

def _potential_grad(x, N):
    """
    Gradient of the potential.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise gradient
    # E_pair = sum_{i<j} d_ij^{-6}
    # d_ij = ||c_i - c_j||
    # dE/dc_i = sum_{j != i} -6 d_ij^{-7} * (c_i - c_j) / d_ij = -6 sum_{j != i} (c_i - c_j) / d_ij^8
    # Wait, derivative of d^{-6} wrt d is -6 d^{-7}.
    # d/dc_i d_ij = (c_i - c_j) / d_ij
    # So dE/dc_i = sum_{j != i} -6 d_ij^{-7} * (c_i - c_j) / d_ij = -6 sum_{j != i} (c_i - c_j) / d_ij^8
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    
    # dist^8 = (dist^2)^4
    dist_8 = dist_sq**4
    
    # Gradient contribution from pair i,j to center i is -6 * (c_i - c_j) / dist_8
    # The matrix G has elements G[i,j] = -6 * diff[i,j] / dist_8[i,j]
    # The total gradient for center i is sum_j G[i,j]
    
    # diff shape (N, N, 3)
    # dist_8 shape (N, N)
    # We need to multiply diff by 1/dist_8 broadcasted
    inv_dist_8 = 1.0 / dist_8
    # Set diagonal to 0 to avoid self-interaction in gradient sum
    np.fill_diagonal(inv_dist_8, 0.0)
    
    # grad_pair[i] = -6 * sum_j (c_i - c_j) * inv_dist_8[i,j]
    # Using einsum: sum over j
    # diff is (N, N, 3), inv_dist_8 is (N, N)
    # We want sum over axis 1 (j)
    grad_pair = -6.0 * np.einsum('ijk,ij->ik', diff, inv_dist_8)
    
    # Wall gradient
    # E_wall = sum_k 1/w_k^6
    # dE/dx = -6/w^7 * dx/dx
    # w = 0.5 - |x|
    # dx/dx = -sign(x)
    # So dE/dx = -6/w^7 * (-sign(x)) = 6 * sign(x) / w^7
    
    centers_flat = centers.reshape(-1)
    signs = np.sign(centers_flat)
    wall_dists_vec = 0.5 - np.abs(centers_flat)
    wall_dists_safe = np.maximum(wall_dists_vec, 1e-6)
    
    # grad_wall_flat[i] = 6 * sign(x_i) / w_i^7
    grad_wall_flat = 6.0 * signs / (wall_dists_safe**7)
    
    # Reshape grad_wall_flat to (N, 3)
    grad_wall = grad_wall_flat.reshape(N, 3)
    
    total_grad = grad_pair + grad_wall
    return total_grad.reshape(-1)

def _calculate_radius(centers):
    """
    Calculate the actual packing radius for a set of centers.
    """
    if centers.shape[0] != N:
        return 0.0
        
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_pair_dist = np.sqrt(min_dist_sq)
    
    # Wall distances
    # dist to wall = 0.5 - |coord|
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].flatten()
    
    # Add small random noise to break symmetry
    x0 = x0 + np.random.normal(0, 0.01, x0.size)
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_radius = _calculate_radius(x0.reshape(N, 3))
    best_x = x0
    
    # Try to optimize
    # We run L-BFGS-B
    # We might run it multiple times with different initializations or noise levels if time permits
    
    attempts = 0
    max_attempts = 5
    
    while attempts < max_attempts and time.time() < deadline:
        try:
            res = minimize(
                _potential, 
                x0, 
                jac=_potential_grad, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_opt = res.x
            # Ensure validity (clip to bounds just in case, though L-BFGS-B should respect them)
            x_opt = np.clip(x_opt, -0.5, 0.5)
            
            r_opt = _calculate_radius(x_opt.reshape(N, 3))
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_x = x_opt
                
            # Perturb the best solution for the next attempt
            # Or use the current result as the next start point?
            # Usually, restarting from a perturbed version of the best found so far is good.
            x0 = best_x + np.random.normal(0, 0.05, best_x.size)
            
        except Exception as e:
            pass
            
        attempts += 1
        
    # Final polish: If we have time, do a tighter optimization on the best found
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                _potential, 
                best_x, 
                jac=_potential_grad, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_opt = np.clip(res.x, -0.5, 0.5)
            r_opt = _calculate_radius(x_opt.reshape(N, 3))
            if r_opt > best_radius:
                best_x = x_opt
        except Exception:
            pass

    return best_x.reshape(N, 3)

# EVOLVE-BLOCK-END
