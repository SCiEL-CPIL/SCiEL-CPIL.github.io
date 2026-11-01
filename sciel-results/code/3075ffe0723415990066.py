import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    # Estimate spacing
    # Area of unit circle = pi. Area of N circles = N * pi * r^2.
    # r ~ 1/sqrt(N). Spacing ~ 2r.
    # For N=50, r ~ 0.14. Spacing ~ 0.28.
    # Let's generate a grid and filter.
    
    def get_hex_init():
        spacing = 0.25
        points = []
        # Generate grid large enough
        k = 10
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        # Sort by distance to origin to pick central ones? 
        # Or just take first N? 
        # Better to take the ones closest to the center to ensure they are inside.
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        return points[indices[:N]]
    
    c = get_hex_init()
    
    # Ensure finite and valid
    if not np.all(np.isfinite(c)) or len(c) != N:
         return np.zeros((N, 2))

    # 2. Soft-Min Objective and Gradient
    def soft_min_obj_and_grad(c, beta):
        c = c.reshape(N, 2)
        # Compute pairwise distances
        # dist[i, j] = |c_i - c_j|
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Boundary distances
        boundary_dist = 1.0 - np.linalg.norm(c, axis=1)
        
        # Values for soft min
        # We need all pairwise i<j distances / 2
        # And all boundary distances
        # To avoid double counting or self-distance, we take upper triangle
        iu = np.triu_indices(N, k=1)
        pair_dists = dist[iu] / 2.0
        
        # Combine all values
        # u = [pair_dists, boundary_dist]
        u = np.concatenate([pair_dists, boundary_dist])
        
        # Numerical stability for soft min
        u_min = np.min(u)
        # If u_min is very negative, exp(-beta * (u - u_min)) might overflow if beta is large and u-u_min is large positive?
        # No, u - u_min >= 0. exp(-beta * positive) <= 1. Safe.
        # But if beta * (u - u_min) is large, exp underflows to 0. That's fine.
        
        exp_terms = np.exp(-beta * (u - u_min))
        sum_exp = np.sum(exp_terms)
        
        # Soft min value
        val = u_min - (1.0 / beta) * np.log(sum_exp)
        
        # We want to maximize val, so minimize -val
        obj = -val
        
        # Weights
        w = exp_terms / sum_exp
        
        # Gradient
        grad = np.zeros_like(c)
        
        # Pairwise part
        # w_pair corresponds to pair_dists
        w_pair = w[:len(pair_dists)]
        
        # Accumulate gradients for pairs
        # For each pair (i, j), weight w_k
        # d/dc_i (dist_ij / 2) = 0.5 * (c_i - c_j) / dist_ij
        # d/dc_j (dist_ij / 2) = 0.5 * (c_j - c_i) / dist_ij
        
        # We can use a matrix approach or loop. N=50 is small, loop is fine but vectorized is better.
        # Let's create a weight matrix W of size N x N, zero diagonal.
        W = np.zeros((N, N))
        # iu[0] are row indices, iu[1] are col indices
        # pair_dists[k] corresponds to (iu[0][k], iu[1][k])
        W[iu[0], iu[1]] = w_pair
        W[iu[1], iu[0]] = w_pair
        
        # dist matrix is symmetric.
        # Avoid division by zero
        dist_safe = np.where(dist == 0, 1e-10, dist)
        
        # grad_i = sum_j W[i,j] * 0.5 * (c_i - c_j) / dist_safe[i,j]
        # This is a weighted sum of vectors.
        
        # Vectorized calculation:
        # diff[i,j] = c_i - c_j
        # We want sum_j W[i,j] * 0.5 * diff[i,j] / dist_safe[i,j]
        
        # Create factor matrix F[i,j] = W[i,j] * 0.5 / dist_safe[i,j]
        F = W * 0.5 / dist_safe
        # Set diagonal to 0 just in case
        np.fill_diagonal(F, 0)
        
        # grad = F @ diff ?
        # (F @ diff)[i] = sum_j F[i,j] * diff[i,j]
        # diff is 3D. We need to sum over j for each i.
        # np.einsum('ij,ijk->ik', F, diff)
        
        grad_pair = np.einsum('ij,ijk->ik', F, diff)
        
        # Boundary part
        w_bound = w[len(pair_dists):]
        # d/dc_i (1 - |c_i|) = -c_i / |c_i|
        c_norm = np.linalg.norm(c, axis=1, keepdims=True)
        c_norm_safe = np.where(c_norm == 0, 1e-10, c_norm)
        grad_bound = -w_bound[:, np.newaxis] * (c / c_norm_safe)
        
        grad = grad_pair + grad_bound
        
        # Since we are minimizing obj = -val, the gradient of obj is -gradient of val.
        # The code above computed d(val)/d(c).
        # So grad_obj = -grad
        return obj, -grad

    # 3. Optimization Loop with Annealing
    betas = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    best_c = c.copy()
    best_score = -np.inf
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Optimize
        try:
            res = minimize(
                soft_min_obj_and_grad, 
                c, 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            c = res.x
            # Check validity
            if np.all(np.isfinite(c)):
                # Calculate actual score (min distance)
                dist = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
                iu = np.triu_indices(N, k=1)
                min_pair = np.min(dist[iu]) / 2.0
                min_bound = np.min(1.0 - np.linalg.norm(c, axis=1))
                current_r = min(min_pair, min_bound)
                
                if current_r > best_score:
                    best_score = current_r
                    best_c = c.copy()
        except Exception:
            pass

    # Final polish with very high beta if time permits
    if time.time() < deadline - 5.0:
         beta = 10000
         try:
             res = minimize(
                soft_min_obj_and_grad, 
                best_c, 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
             c = res.x
             if np.all(np.isfinite(c)):
                dist = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
                iu = np.triu_indices(N, k=1)
                min_pair = np.min(dist[iu]) / 2.0
                min_bound = np.min(1.0 - np.linalg.norm(c, axis=1))
                current_r = min(min_pair, min_bound)
                
                if current_r > best_score:
                    best_score = current_r
                    best_c = c.copy()
         except Exception:
             pass

    return best_c
