import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    def energy_and_grad(x):
        centers = x.reshape(N, 3)
        
        # Calculate pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e-9)
        
        # Energy term 1: Sphere-Sphere repulsion
        # Use 1/d^2 to be smooth and strongly repulsive when close
        # Gradient: d/dc_i (1/d_ij^2) = -2 (c_i - c_j) / d_ij^4
        # Sum over j
        d4 = d2**2
        np.fill_diagonal(d4, 1e-9)
        
        # force_ij = -2 * diff[i,j] / d4[i,j]
        # Total force on i = sum_j force_ij
        # Note: diff is c_i - c_j.
        # Gradient of energy wrt c_i is sum_{j != i} -2 (c_i - c_j) / d_{ij}^4
        # We want to minimize energy, so we move in direction of -gradient.
        # -grad = sum_{j != i} 2 (c_i - c_j) / d_{ij}^4
        # This is a repulsive force.
        
        # Let's compute the gradient of the energy function directly.
        # E_ss = sum_{i<j} 1/d_ij^2
        # dE_ss/dc_i = sum_{j != i} -2 (c_i - c_j) / d_ij^4
        
        grad_ss = np.zeros_like(centers)
        # Use the diff matrix
        # factor = -2 / d4
        factor = -2.0 / d4
        np.fill_diagonal(factor, 0)
        
        # grad_ss[i] = sum_j factor[i,j] * diff[i,j]
        # diff[i,j] is c_i - c_j
        grad_ss = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Energy term 2: Wall penalty
        # E_wall = sum_{i,k} 1 / (0.5 - |c_ik|)^2  ... this is bad near 0.5
        # Better: Penalize distance to wall.
        # Let w_ik = 0.5 - |c_ik|.
        # If w_ik < 0, it's outside.
        # Use a smooth penalty: E_wall = sum_{i,k} max(0, 0.5 - |c_ik| - r_target)^2 ? No, we don't know r_target.
        # Standard approach: E_wall = sum_{i,k} 1 / (0.5 - |c_ik| + eps)^2
        # This blows up as c approaches 0.5.
        
        # Let's use a softer wall potential:
        # E_wall = sum_{i,k} (max(0, |c_ik| - 0.5 + eps))^2 is for hard walls.
        # We want to keep them inside.
        # Let's define distance to wall d_w = 0.5 - |c_ik|.
        # If d_w > 0, no penalty? No, we want to maximize min d_w.
        # So we should minimize -min d_w? No, that's non-smooth.
        # Minimize sum 1/d_w^2.
        
        # Let's compute wall gradient.
        # E_w = sum_{i,k} (0.5 - |c_ik|)^{-2}
        # dE_w/dc_ik = -2 (0.5 - |c_ik|)^{-3} * d(0.5 - |c_ik|)/dc_ik
        # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
        # So dE_w/dc_ik = 2 (0.5 - |c_ik|)^{-3} * sign(c_ik)
        
        # This gradient is undefined at c=0. We can add a small epsilon or use abs(c) derivative carefully.
        # Actually, for c=0, sign is 0, so gradient is 0. This is a local minimum of the penalty?
        # E(c) = (0.5 - |c|)^-2. At c=0, E = 4. At c=0.1, E = (0.4)^-2 = 6.25.
        # So c=0 is a minimum of the penalty (best position wrt walls).
        # The gradient should point towards 0.
        # d/dc (0.5 - |c|)^-2.
        # For c > 0: d/dc (0.5 - c)^-2 = -2(0.5-c)^-3 * (-1) = 2(0.5-c)^-3. Positive.
        # So gradient is positive for c>0. Minimizing moves c to smaller values (towards 0). Correct.
        # For c < 0: d/dc (0.5 + c)^-2 = -2(0.5+c)^-3 * (1) = -2(0.5+c)^-3. Negative.
        # So gradient is negative for c<0. Minimizing moves c to larger values (towards 0). Correct.
        
        grad_wall = np.zeros_like(centers)
        for k in range(3):
            c = centers[:, k]
            dist = 0.5 - np.abs(c)
            # Avoid division by zero if c is exactly 0.5 (shouldn't happen with bounds)
            dist = np.maximum(dist, 1e-6)
            # grad = 2 * dist^-3 * sign(c)
            signs = np.sign(c)
            # Handle c=0: sign is 0, grad is 0.
            grad_wall[:, k] = 2.0 * (dist ** -3) * signs
            
        total_grad = grad_ss + grad_wall
        
        # Calculate energy value
        # E_ss = sum_{i<j} 1/d2
        e_ss = 0.5 * np.sum(1.0 / d2) # 0.5 because sum i<j is half of sum i!=j
        # E_wall = sum 1/dist^2
        e_wall = np.sum(1.0 / (dist**2)) # dist is 0.5 - |c|
        
        return e_ss + e_wall, total_grad.flatten()

    def run_optimization(init_centers):
        x0 = init_centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        res = minimize(
            lambda x: energy_and_grad(x)[0],
            x0,
            jac=lambda x: energy_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        best_centers = res.x.reshape(N, 3)
        
        # Calculate radius
        diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e9)
        d_min = np.sqrt(np.min(d2))
        
        w_min = np.min(0.5 - np.abs(best_centers))
        
        r = min(d_min / 2.0, w_min)
        
        return best_centers, r

    def get_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e9)
        d_min = np.sqrt(np.min(d2))
        w_min = np.min(0.5 - np.abs(centers))
        return min(d_min / 2.0, w_min)

    start_time = time.time()
    time_limit = 110.0
    
    best_r = 0.0
    best_centers = None
    
    # Multiple restarts
    for i in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Generate initial configuration
        # Mix of random and jittered grid
        if i % 2 == 0:
            # Jittered Grid
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            grid = grid[:N]
            jitter = np.random.normal(0, 0.05, grid.shape)
            init = np.clip(grid + jitter, -0.49, 0.49)
        else:
            # Random
            init = np.random.uniform(-0.4, 0.4, (N, 3))
            
        centers, r = run_optimization(init)
        
        if r > best_r:
            best_r = r
            best_centers = centers
            
    # Final polish on the best one
    if time.time() - start_time < time_limit and best_centers is not None:
        # Run optimization again on the best result to refine
        centers, r = run_optimization(best_centers)
        if r > best_r:
            best_r = r
            best_centers = centers

    return best_centers
