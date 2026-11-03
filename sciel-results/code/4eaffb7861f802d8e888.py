import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    max_time = 110.0  # Safety margin for 120s limit
    
    best_radius = 0.0
    best_centers = None
    
    # Initial grid for fallback or initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    best_radius = _calculate_radius(best_centers)
    
    def repulsion_energy(x, power=2.0):
        """
        x: shape (N, 3)
        Returns repulsion energy and gradient.
        """
        x = x.reshape(N, 3)
        energy = 0.0
        grad = np.zeros_like(x)
        
        # Pairwise repulsion
        # d_ij^2 = ||x_i - x_j||^2
        # We use 1/d^power. Let's stick to power=2 for stability and speed.
        # Actually, higher power focuses more on the minimum distance.
        # Let's use a soft min approximation or just sum of 1/d^2.
        
        # Efficient pairwise distance calculation
        # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # This creates an N x N x 3 array, which is 100x100x3 = 30000 floats. Very manageable.
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero for i=j
        np.fill_diagonal(dist_sq, 1.0)
        
        # We want to minimize sum(1/dist^power).
        # Let power = 2. term = 1/dist_sq.
        # grad of 1/dist_sq wrt x_i:
        # d/dx_i (1/||x_i - x_j||^2) = -2 (x_i - x_j) / ||x_i - x_j||^4
        # Sum over j.
        
        # term = 1.0 / dist_sq
        # energy += np.sum(term)
        # 
        # grad contribution for i from j:
        # -2 * diff[i,j] / dist_sq[i,j]**2
        # 
        # Let's compute this.
        inv_dist_sq = 1.0 / dist_sq
        # energy += np.sum(inv_dist_sq)
        
        # Gradient calculation
        # grad_i = sum_j ( -2 * (x_i - x_j) / dist_sq_ij^2 )
        # Note: diff[i,j] = x_i - x_j
        # So we need -2 * diff * inv_dist_sq^2
        # Sum over j (axis=1)
        
        # mask diagonal to be 0
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq[~mask] = 0.0 # zero out diagonal
        
        # term_grad = -2 * diff * (inv_dist_sq[:, :, np.newaxis] ** 2)
        # This is N x N x 3.
        # Sum over axis 1 (j) to get grad for each i.
        
        # Wait, the energy is sum_{i<j} 1/d_ij^2.
        # My current sum includes i!=j, so it's 2 * sum_{i<j}.
        # This is fine for minimization, just scales the gradient.
        
        # Let's be careful with the gradient.
        # f(x) = sum_{i!=j} (1/||x_i - x_j||^2)
        # df/dx_i = sum_{j!=i} -2 (x_i - x_j) / ||x_i - x_j||^4
        
        # inv_dist_sq is 1/d^2.
        # We need 1/d^4 = (1/d^2)^2.
        inv_dist_sq_sq = inv_dist_sq ** 2
        
        # grad_i = -2 * sum_j (x_i - x_j) * inv_dist_sq_sq[i,j]
        # diff is (x_i - x_j).
        # So grad = -2 * sum_j diff[i,j] * inv_dist_sq_sq[i,j]
        
        # Multiply diff by inv_dist_sq_sq expanded
        # diff: (N, N, 3)
        # inv_dist_sq_sq: (N, N)
        # We want to multiply each vector diff[i,j] by scalar inv_dist_sq_sq[i,j]
        # Then sum over j (axis 1).
        
        grad_pairwise = -2.0 * np.sum(diff * inv_dist_sq_sq[:, :, np.newaxis], axis=1)
        
        energy += np.sum(inv_dist_sq)
        grad += grad_pairwise
        
        # Wall repulsion
        # Distance to walls:
        # x in [-0.5, 0.5].
        # dist_x = min(x + 0.5, 0.5 - x)
        # dist_y = min(y + 0.5, 0.5 - y)
        # dist_z = min(z + 0.5, 0.5 - z)
        
        # To make it smooth, we can use -log or 1/d^2.
        # Let's use 1/d^2 for consistency.
        # d_x = min(x+0.5, 0.5-x)
        # If x < 0, d_x = x+0.5, grad = 1
        # If x > 0, d_x = 0.5-x, grad = -1
        # This is non-differentiable at x=0.
        # A smooth approximation is needed, or we can use a potential that is smooth.
        # Alternatively, we can just project back to the box after each step if using gradient descent,
        # but L-BFGS-B handles bounds.
        
        # Let's use bounds in L-BFGS-B instead of wall repulsion energy.
        # This is more robust.
        
        return energy, grad

    def objective(x):
        energy, grad = repulsion_energy(x)
        return energy, grad

    bounds = [(-0.5, 0.5)] * (N * 3)

    # Multiple restarts
    num_restarts = 50
    rng = np.random.default_rng(42)
    
    for i in range(num_restarts):
        if time.time() - start_time > max_time - 5:
            break
            
        # Initialization
        if i == 0:
            x0 = best_centers.flatten()
        else:
            # Random initialization inside the cube
            # Jitter the grid or pure random?
            # Pure random might be too chaotic. Grid + noise is better.
            x0 = best_centers + rng.normal(0, 0.1, (N, 3))
            # Ensure within bounds
            x0 = np.clip(x0, -0.49, 0.49)
            x0 = x0.flatten()
            
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < float('inf'):
                x_opt = res.x.reshape(N, 3)
                # Clip to ensure strictly inside (due to floating point errors)
                x_opt = np.clip(x_opt, -0.499999, 0.499999)
                
                r = _calculate_radius(x_opt)
                if r > best_radius:
                    best_radius = r
                    best_centers = x_opt.copy()
                    
        except Exception:
            continue

    return best_centers

def _calculate_radius(centers):
    """
    Calculate the maximum radius for the given centers.
    r = min( min_{i<j} ||c_i - c_j||/2, min_i min_k (0.5 - |c_ik|) )
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, float('inf'))
    min_pair_dist = np.sqrt(np.min(dist_sq)) / 2.0
    
    # Wall distances
    # centers in [-0.5, 0.5]
    # dist to wall = min(c + 0.5, 0.5 - c)
    # Since c is in [-0.5, 0.5], c+0.5 >= 0 and 0.5-c >= 0.
    # min(c+0.5, 0.5-c) is equivalent to 0.5 - |c|
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist)
