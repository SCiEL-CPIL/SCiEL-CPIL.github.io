import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # seconds, leave 20s margin for overhead/return
    
    # 1. Initialization: Hexagonal Grid
    # Generate a hexagonal lattice
    # Spacing a
    a = 0.07 
    # Generate points in a box larger than the unit square
    # x from -0.6 to 0.6, y from -0.6 to 0.6
    x_min, x_max = -0.6, 0.6
    y_min, y_max = -0.6, 0.6
    
    points = []
    # Hex grid: 
    # Row j: y = j * a * sqrt(3)
    # If j even: x = i * a
    # If j odd: x = i * a + a/2
    j_max = int((y_max - y_min) / (a * np.sqrt(3))) + 2
    i_max = int((x_max - x_min) / a) + 2
    
    for j in range(-j_max, j_max + 1):
        y = j * a * np.sqrt(3)
        if y < y_min or y > y_max:
            continue
        for i in range(-i_max, i_max + 1):
            x = i * a
            if j % 2 != 0:
                x += a / 2.0
            if x < x_min or x > x_max:
                continue
            # Filter to unit square [-0.5, 0.5]
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append((x, y))
                
    points = np.array(points)
    
    # If we have more than N points, select the N closest to the origin
    if len(points) > N:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)
        points = points[idx[:N]]
    elif len(points) < N:
        # Fallback: Square grid if hex grid failed to provide enough points
        # This is unlikely with a=0.07, but safe
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
        
    # Ensure shape is (N, 2)
    points = points.reshape(-1, 2)
    
    # 2. Optimization
    # Objective: Maximize soft-min of distances
    # Distances:
    # - Pairwise: |c_i - c_j| / 2
    # - Wall: 0.5 - |c_i_k|
    
    def compute_distances(x):
        # x is (N, 2)
        c = x
        # Pairwise distances
        # Use broadcasting to get all pairwise distances
        # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists_sq, np.inf)
        # We only need upper triangle or all, soft-min handles it
        # Taking sqrt
        pair_dists = np.sqrt(dists_sq)
        # Divide by 2
        pair_dists /= 2.0
        
        # Wall distances
        # 0.5 - |x|, 0.5 - |y|
        wall_dists_x = 0.5 - np.abs(c[:, 0])
        wall_dists_y = 0.5 - np.abs(c[:, 1])
        
        # Combine into a flat array
        # Flatten pair_dists (N*N) and wall_dists (2*N)
        # Note: pair_dists has inf on diagonal, which will not contribute to exp(-beta * inf) = 0
        all_dists = np.concatenate([pair_dists.flatten(), wall_dists_x, wall_dists_y])
        return all_dists

    def objective_and_grad(x, beta):
        dists = compute_distances(x)
        # Soft min: S = -1/beta * log(sum(exp(-beta * d)))
        # To avoid underflow/overflow, use log-sum-exp trick
        # log(sum(exp(-beta * d))) = -beta * max(d) + log(sum(exp(-beta * (d - max(d)))))
        # Wait, we want min(d). 
        # min(d) = - max(-d).
        # Let v = -d. We want max(v).
        # max(v) = log(sum(exp(v))) - log(count) approx? No.
        # Softmax max: M = -1/beta * log(sum(exp(-beta * v))) where v = -d?
        # Let's stick to: S = -1/beta * log(sum(exp(-beta * d)))
        # This approximates min(d).
        
        # Compute -beta * d
        exp_args = -beta * dists
        # Subtract max for stability
        max_arg = np.max(exp_args)
        exp_args_shifted = exp_args - max_arg
        sum_exp = np.sum(np.exp(exp_args_shifted))
        log_sum_exp = max_arg + np.log(sum_exp)
        S = -log_sum_exp / beta
        
        # Gradient of S wrt x
        # dS/dx = -1/beta * d(log_sum_exp)/dx
        # d(log_sum_exp)/dx = sum(exp(exp_args_shifted) / sum_exp) * d(exp_args)/dx
        # d(exp_args)/dx = -beta * d(dists)/dx
        # So dS/dx = -1/beta * (weights * (-beta * grad_dists)) = weights * grad_dists
        # weights = exp(exp_args_shifted) / sum_exp
        
        weights = np.exp(exp_args_shifted) / sum_exp
        
        # Now we need grad_dists wrt x
        # dists is a concatenation of pair_dists and wall_dists
        # Let's compute gradients separately and combine
        
        c = x
        N_c = c.shape[0]
        
        # Wall gradients
        # d(0.5 - |x_i|)/dx_i = -sign(x_i)
        # d(0.5 - |y_i|)/dy_i = -sign(y_i)
        grad_wall = np.zeros_like(c)
        # wall_dists_x corresponds to indices N_c*N_c to N_c*N_c + N_c - 1
        # wall_dists_y corresponds to indices N_c*N_c + N_c to N_c*N_c + 2*N_c - 1
        
        w_x = weights[N_c*N_c : N_c*N_c + N_c]
        w_y = weights[N_c*N_c + N_c : N_c*N_c + 2*N_c]
        
        grad_wall[:, 0] -= np.sign(c[:, 0]) * w_x
        grad_wall[:, 1] -= np.sign(c[:, 1]) * w_y
        
        # Pair gradients
        # dist = 0.5 * |c_i - c_j|
        # d(dist)/dc_i = 0.5 * (c_i - c_j) / |c_i - c_j|
        # d(dist)/dc_j = 0.5 * (c_j - c_i) / |c_i - c_j|
        
        # We have weights for all pair distances in the first N_c*N_c elements
        w_pair = weights[:N_c*N_c].reshape(N_c, N_c)
        
        # Compute differences and distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_vec = np.sqrt(np.sum(diff**2, axis=2))
        # Avoid division by zero
        dists_vec_safe = np.where(dists_vec > 1e-10, dists_vec, 1.0)
        
        # Unit vectors
        unit_vecs = diff / dists_vec_safe[:, :, np.newaxis]
        
        # Gradient contribution from pair (i, j) to c_i is 0.5 * unit_vec[i, j] * w_pair[i, j]
        # Gradient contribution from pair (i, j) to c_j is 0.5 * (-unit_vec[i, j]) * w_pair[i, j]
        # Sum over all j for each i
        
        # grad_pair[i] = sum_j 0.5 * unit_vec[i, j] * w_pair[i, j]
        # Note: w_pair is symmetric if dists are symmetric, which they are.
        # But we computed all pairs.
        
        grad_pair = np.zeros_like(c)
        # Vectorized sum
        # For each i, sum over j of w_pair[i, j] * unit_vec[i, j]
        # This is a matrix product if we view it appropriately?
        # grad_pair = 0.5 * (w_pair * unit_vecs).sum(axis=1)
        
        # w_pair shape (N, N), unit_vecs shape (N, N, 2)
        # Multiply w_pair[:, :, np.newaxis] with unit_vecs
        weighted_units = w_pair[:, :, np.newaxis] * unit_vecs
        grad_pair = 0.5 * np.sum(weighted_units, axis=1)
        
        total_grad = grad_wall + grad_pair
        
        # We want to minimize -S
        val = -S
        grad = -total_grad
        
        return val, grad

    # Optimization Loop
    # We will run L-BFGS-B
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    x_current = points.copy()
    
    # Try different beta values
    # Start with lower beta for global-ish search, then increase
    betas = [20, 50, 100, 200]
    
    best_val = np.inf
    best_x = x_current
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Run minimize
        # maxiter is a safety, but we rely on time
        res = minimize(
            objective_and_grad, 
            x_current, 
            args=(beta,), 
            method='L-BFGS-B', 
            jac=True,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        x_current = res.x
        
        # Check time
        if time.time() > deadline:
            break
            
        # Calculate actual score for this configuration
        # Score = min(min_pair_dist/2, min_wall_dist)
        c = x_current
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dists_sq)) / 2.0
        
        wall_dists_x = 0.5 - np.abs(c[:, 0])
        wall_dists_y = 0.5 - np.abs(c[:, 1])
        min_wall_dist = min(np.min(wall_dists_x), np.min(wall_dists_y))
        
        current_score = min(min_pair_dist, min_wall_dist)
        
        if current_score > best_val:
            best_val = current_score
            best_x = x_current.copy()
            
    # Final Polish: If time remains, do a simple local search or one more high-beta iteration
    if time.time() < deadline - 10:
        # One more iteration with high beta
        beta_final = 500
        try:
            res = minimize(
                objective_and_grad, 
                best_x, 
                args=(beta_final,), 
                method='L-BFGS-B', 
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_final = res.x
            
            # Verify
            c = x_final
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            min_pair_dist = np.sqrt(np.min(dists_sq)) / 2.0
            
            wall_dists_x = 0.5 - np.abs(c[:, 0])
            wall_dists_y = 0.5 - np.abs(c[:, 1])
            min_wall_dist = min(np.min(wall_dists_x), np.min(wall_dists_y))
            
            final_score = min(min_pair_dist, min_wall_dist)
            
            if final_score > best_val:
                best_x = x_final
                best_val = final_score
        except Exception:
            pass
            
    # Ensure validity
    # Clip to bounds just in case
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x

# EVOLVE-BLOCK-END
