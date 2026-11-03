"""
Strategy: Physics-Informed Optimization using Repulsive Potentials.
Instead of optimizing the non-differentiable min-distance directly, we minimize a smooth
repulsive potential energy function. This allows the use of efficient gradient-based
optimizers (L-BFGS-B) which can navigate the high-dimensional landscape (300 variables)
much faster than black-box methods like Simulated Annealing for this specific problem size.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # 1. Initial Guess: Use a jittered grid to break symmetry and provide a good starting point
    # A 5x5x5 grid is exactly 125 points, but we only need 100.
    # We can start with a 4x5x5 = 100 grid or 5x5x5 and remove 25, or just use a 4x5x5 grid directly.
    # Let's use a 4x5x5 grid for exactly 100 points.
    nx, ny, nz = 4, 5, 5
    if nx * ny * nz < N:
        # Fallback if N changes (though N is fixed at 100)
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        x0 = grid[:N].copy()
    else:
        # Create 4x5x5 grid
        x_coords = (np.arange(nx) + 0.5) / nx - 0.5
        y_coords = (np.arange(ny) + 0.5) / ny - 0.5
        z_coords = (np.arange(nz) + 0.5) / nz - 0.5
        grid = np.array([(x, y, z) for z in z_coords for y in y_coords for x in x_coords])
        x0 = grid[:N].copy()
    
    # Add small random jitter to break degeneracy
    np.random.seed(42)
    x0 += np.random.uniform(-0.01, 0.01, x0.shape)
    # Ensure inside bounds
    x0 = np.clip(x0, -0.49, 0.49)

    # 2. Define Objective Function: Sum of Repulsive Potentials
    # We want to maximize the minimum distance. Minimizing the sum of 1/d^p (with large p)
    # approximates maximizing the min distance while keeping the function smooth.
    # However, 1/d^p is singular at d=0. We use a softening parameter epsilon.
    # E = sum_{i<j} 1 / (d_ij^2 + eps)^k
    # Gradient descent on this pushes all points apart, but "closest" pairs contribute most to the gradient.
    
    def energy_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        # Compute pairwise distances
        # x_i - x_j
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal (self-distance)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Softening to avoid division by zero and handle very close points smoothly
        # We use a "soft min" approximation. 
        # Actually, a better surrogate for max-min distance is the negative of the "soft min" of distances.
        # But repulsive energy is standard for packing.
        # Let's use E = sum 1/(d^2 + eps)
        # To make it focus more on the minimum distance, we can increase the power.
        # E = sum 1/(d^2 + eps)^alpha
        # alpha=1 is standard Coulomb. Higher alpha focuses more on the closest pair.
        alpha = 2.0 
        eps = 1e-6
        
        # d^2 + eps
        denom = dist_sq + eps
        # We only need upper triangle to avoid double counting, but for gradient symmetry it's easier to sum all and divide by 2
        # Or just sum all unique pairs.
        # Let's compute the sum for all i != j
        # term = 1 / denom^alpha
        term = denom ** (-alpha)
        np.fill_diagonal(term, 0.0)
        
        energy = np.sum(term) / 2.0
        
        # Gradient
        # d/dx_i [ 1 / (|x_i - x_j|^2 + eps)^alpha ]
        # = -alpha * (|x_i - x_j|^2 + eps)^(-alpha - 1) * 2 * (x_i - x_j)
        # Sum over all j
        # grad_i = sum_j -2 * alpha * (dist_sq_ij + eps)^(-alpha - 1) * (x_i - x_j)
        
        # We need (dist_sq + eps)^(-alpha - 1)
        grad_factor = (-2.0 * alpha) * (denom ** (-alpha - 1))
        np.fill_diagonal(grad_factor, 0.0)
        
        # grad_i = sum_j grad_factor[i,j] * (x_i - x_j)
        # diff[i,j] = x_i - x_j
        # grad = sum_j (grad_factor[i,j] * diff[i,j])
        grad = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Reshape gradient
        grad_flat = grad.flatten()
        
        return energy, grad_flat

    # 3. Optimization Loop with Restarts
    best_x = x0.copy()
    best_score = -np.inf
    
    # Helper to calculate the actual score (min distance / 2)
    def calculate_score(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        
        # Boundary constraint check
        min_coord = np.min(x, axis=0)
        max_coord = np.max(x, axis=0)
        boundary_dist = min(0.5 - min_coord, max_coord + 0.5) # min(0.5 - min, 0.5 - (-max)) -> min(0.5-min, 0.5+max)
        # Actually: min_i min_k (1/2 - |c_ik|)
        # This is equivalent to min( 0.5 - max(x), 0.5 - min(x) ) for each dimension? No.
        # It is min over all points and dimensions of (0.5 - |x_k|)
        # Which is 0.5 - max(|x|)
        max_abs = np.max(np.abs(x))
        boundary_dist = 0.5 - max_abs
        
        r = min(min_dist / 2.0, boundary_dist)
        return r

    # Run L-BFGS-B
    # We will run a few iterations/restarts if time permits
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # If not the first restart, perturb the best solution found so far
        if restart > 0:
            current_x = best_x + np.random.uniform(-0.05, 0.05, best_x.shape)
            current_x = np.clip(current_x, -0.49, 0.49)
        else:
            current_x = x0.copy()
        
        try:
            # Use L-BFGS-B with bounds
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            # Limit iterations to avoid hanging
            maxiter = 500
            
            res = minimize(
                energy_and_grad, 
                current_x.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': maxiter, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.success or res.fun < np.inf:
                new_x = res.x.reshape(-1, 3)
                new_score = calculate_score(new_x)
                
                if new_score > best_score:
                    best_score = new_score
                    best_x = new_x.copy()
                    
        except Exception as e:
            pass # Ignore optimization failures and continue

    # Final Polish:
    # Sometimes the gradient method gets stuck in local minima where points are "close" but not optimal.
    # We can try a simple local search on the best few pairs if time remains.
    if time.time() < deadline - 2.0:
        # Identify the pair with minimum distance
        diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        min_val = np.min(dist)
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        
        # Try to move i and j apart slightly
        # Direction from j to i
        dir_vec = best_x[i] - best_x[j]
        norm = np.linalg.norm(dir_vec)
        if norm > 1e-9:
            dir_vec /= norm
            
            # Try small perturbations
            for step in [0.01, 0.005, 0.001]:
                trial_x = best_x.copy()
                trial_x[i] += dir_vec * step
                trial_x[j] -= dir_vec * step
                # Check bounds
                if np.all(np.abs(trial_x) < 0.5):
                    trial_score = calculate_score(trial_x)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_x = trial_x

    # Ensure valid output
    # Clip to just inside the cube to avoid floating point issues with boundary
    best_x = np.clip(best_x, -0.49999999, 0.49999999)
    
    return best_x
