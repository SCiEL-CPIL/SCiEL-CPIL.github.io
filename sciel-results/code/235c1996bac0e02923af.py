import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_points = None
    best_score = -1.0
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    # Wall indices: for each sphere i, distance to 3 walls (x, y, z)
    # We handle walls separately in the gradient calculation for clarity
    
    def soft_min_score(points, beta=100.0):
        """
        Returns the negative soft-min of all pairwise distances (divided by 2) 
        and distances to walls.
        Minimizing this maximizes the soft-min radius.
        """
        # points shape (N, 3)
        # Clip to valid range to ensure finite values, though optimizer should keep them in bounds
        p = np.clip(points, -0.5, 0.5)
        
        # Pairwise distances
        # d_ij = 0.5 * |p_i - p_j|
        diff = p[i_idx] - p[j_idx]
        dists = 0.5 * np.sqrt(np.sum(diff**2, axis=1))
        
        # Wall distances
        # d_i_wall = 0.5 - |p_ik|
        # We can compute these for all 3 axes
        # p is (N, 3)
        # abs_p = |p|
        # dists_wall = 0.5 - abs_p
        # Flatten dists_wall to (N*3,)
        abs_p = np.abs(p)
        dists_wall = 0.5 - abs_p
        dists_wall_flat = dists_wall.flatten()
        
        # Combine all distances
        # dists shape (n_pairs,)
        # dists_wall_flat shape (N*3,)
        all_dists = np.concatenate([dists, dists_wall_flat])
        
        # Soft min
        # To avoid overflow in exp, subtract max
        max_d = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - max_d))
        log_sum = np.log(np.sum(exp_terms)) + beta * max_d
        soft_min_val = log_sum / beta
        
        return -soft_min_val

    def soft_min_grad(points, beta=100.0):
        """
        Gradient of the soft_min_score.
        """
        p = np.clip(points, -0.5, 0.5)
        grad = np.zeros_like(p)
        
        # Pairwise distances
        diff = p[i_idx] - p[j_idx]
        dists = 0.5 * np.sqrt(np.sum(diff**2, axis=1))
        
        # Wall distances
        abs_p = np.abs(p)
        dists_wall = 0.5 - abs_p
        dists_wall_flat = dists_wall.flatten()
        
        all_dists = np.concatenate([dists, dists_wall_flat])
        
        # Weights
        max_d = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - max_d))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Gradient of soft_min = sum w_k * grad d_k
        # grad score = - grad soft_min = - sum w_k * grad d_k
        
        # 1. Pairwise part
        # d_k = 0.5 * |diff_k|
        # grad d_k w.r.t p_i = 0.5 * diff_k / |diff_k| = 0.5 * diff_k / (2 * d_k) = diff_k / (4 * d_k)
        # grad d_k w.r.t p_j = -0.5 * diff_k / |diff_k| = -diff_k / (4 * d_k)
        
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-8, dists, 1e-8)
        coeffs = weights[:n_pairs] / (4.0 * safe_dists)
        
        # Accumulate gradient for pairs
        # For each pair k, grad w.r.t p_i is coeffs[k] * diff[k]
        # grad w.r.t p_j is -coeffs[k] * diff[k]
        # We use np.add.at for efficient accumulation
        np.add.at(grad, i_idx, coeffs[:, np.newaxis] * diff)
        np.add.at(grad, j_idx, -coeffs[:, np.newaxis] * diff)
        
        # 2. Wall part
        # d = 0.5 - |p_ik|
        # grad d w.r.t p_ik = -sign(p_ik)
        # For each sphere i and axis k, weight is w_wall[i*3+k]
        # grad w.r.t p_ik = -w_wall[i*3+k] * sign(p_ik)
        
        # weights[n_pairs:] corresponds to dists_wall_flat
        w_wall = weights[n_pairs:]
        # Reshape to (N, 3)
        w_wall_reshaped = w_wall.reshape(N, 3)
        
        # sign(p) is 1 if p>0, -1 if p<0, 0 if p==0
        # If p==0, gradient is undefined, but we can use 0 or arbitrary.
        # Using sign is fine.
        signs = np.sign(p)
        
        # grad contribution = - w_wall_reshaped * signs
        grad -= w_wall_reshaped * signs
        
        return grad

    # Optimization loop
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initial guess: Random points in the cube
        # Avoid exact boundaries
        x0 = np.random.uniform(-0.4, 0.4, size=(N, 3))
        
        # Bounds for L-BFGS-B
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Flatten x0
        x0_flat = x0.flatten()
        
        try:
            res = minimize(
                soft_min_score, 
                x0_flat, 
                jac=soft_min_grad, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
        except Exception:
            continue
            
        # Reshape result
        points = res.x.reshape(N, 3)
        
        # Calculate actual score (min distance)
        # We need to compute the true min, not the soft min
        diff = points[i_idx] - points[j_idx]
        dists = 0.5 * np.sqrt(np.sum(diff**2, axis=1))
        min_pair_dist = np.min(dists)
        
        abs_p = np.abs(points)
        dists_wall = 0.5 - abs_p
        min_wall_dist = np.min(dists_wall)
        
        score = min(min_pair_dist, min_wall_dist)
        
        if score > best_score:
            best_score = score
            best_points = points
            
        # Optional: Local polish with Nelder-Mead on the raw min function?
        # L-BFGS-B should be sufficient.
        
    if best_points is None:
        # Fallback to grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_points
