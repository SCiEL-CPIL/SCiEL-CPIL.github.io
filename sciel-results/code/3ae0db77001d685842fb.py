import numpy as np
import time

N = 300

def construct():
    # Initialize with a hexagonal lattice inside the unit circle
    s_init = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    points = []
    R = 1.2
    k = int(R / s_init) + 2
    for j in range(-k, k + 1):
        y = j * s_init * np.sqrt(3) / 2
        x_offset = (j % 2) * s_init / 2
        for i in range(-k, k + 1):
            x = i * s_init + x_offset
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        points = np.random.randn(N, 2) * 0.5
    else:
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N]]
        
    x = points.astype(np.float64).copy()
    
    def calc_score(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
        return min(min_pair, min_bound)
    
    best_x = x.copy()
    best_r = calc_score(best_x)
    
    start_time = time.time()
    max_time = 115.0
    
    # Parameters
    beta = 200.0  # Sharpness for log-sum-exp
    alpha = 0.02  # Step size
    alpha_min = 0.001
    
    def compute_smooth_min(x, beta):
        """
        Compute a smooth approximation of min(min_pair_dist/2, min_bound_dist).
        Uses log-sum-exp for the min over all constraints.
        """
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # For pairs, constraint is dist >= 2r -> r <= dist/2
        # We want to maximize the minimum of all (dist_ij / 2) and (1 - |c_i|)
        # Equivalent to minimizing max(2r - dist_ij, 2r - 2(1-|c_i|))
        # Let's use a direct smooth min of the values v_k
        # Values: v_ij = dist_ij / 2 for i<j, v_i = 1 - |c_i|
        
        # Efficiently get lower triangle indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = np.sqrt(dists_sq[i_idx, j_idx])
        pair_vals = pair_dists / 2.0
        
        norms = np.linalg.norm(x, axis=1)
        bound_vals = 1.0 - norms
        
        # Concatenate all constraint values
        vals = np.concatenate([pair_vals, bound_vals])
        
        # Log-sum-exp for min: -logsumexp(-vals * beta) / beta
        # To avoid overflow, subtract max
        vals_scaled = vals * beta
        max_val = np.max(vals_scaled)
        if not np.isfinite(max_val):
            return 0.0
        vals_scaled = vals_scaled - max_val
        logsumexp = np.log(np.sum(np.exp(vals_scaled)))
        smooth_min = (max_val + logsumexp) / beta
        
        return smooth_min

    def compute_smooth_grad(x, beta):
        """
        Compute gradient of the smooth min.
        d/dx (-logsumexp(-beta * vals) / beta) = sum_k (exp(-beta*val_k)/sum(exp(-beta*val_k))) * grad(val_k) * (beta/beta)
        Actually, let f(x) = min_k val_k(x). Smooth approx g(x) = -1/beta * log(sum_k exp(-beta * val_k(x))).
        grad g = sum_k [ exp(-beta * val_k) / sum_j exp(-beta * val_j) ] * grad val_k
        """
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = np.sqrt(np.maximum(dists_sq[i_idx, j_idx], 1e-12))
        # Gradient of dist/2 w.r.t x_i: (x_i - x_j) / (2 * dist)
        # Gradient of dist/2 w.r.t x_j: (x_j - x_i) / (2 * dist)
        
        # Weights for pairs
        # val_k = dist_k / 2
        # -beta * val_k = -beta * dist_k / 2
        log_weights_pair = -beta * pair_dists / 2.0
        
        norms = np.linalg.norm(x, axis=1)
        bound_vals = 1.0 - norms
        log_weights_bound = -beta * bound_vals
        
        all_log_weights = np.concatenate([log_weights_pair, log_weights_bound])
        
        # Normalize weights
        max_lw = np.max(all_log_weights)
        if not np.isfinite(max_lw):
            return np.zeros_like(x)
        
        all_lw = all_log_weights - max_lw
        exp_lw = np.exp(all_lw)
        sum_exp = np.sum(exp_lw)
        
        if sum_exp == 0:
            return np.zeros_like(x)
            
        weights = exp_lw / sum_exp
        
        grad = np.zeros_like(x)
        
        # Pair gradients
        w_pair = weights[:len(i_idx)]
        
        # Vectorized gradient accumulation is complex, so we iterate over significant pairs
        # To be fast, only process pairs with non-negligible weight
        # Since weights decay exponentially, only the smallest distances matter
        # Sort by distance to process only top M
        order = np.argsort(pair_dists)
        M = 200  # Number of closest pairs to consider
        
        for idx in order[:M]:
            w = w_pair[idx]
            if w < 1e-6:
                break
            i, j = i_idx[idx], j_idx[idx]
            d = pair_dists[idx]
            if d < 1e-8:
                continue
            v = x[i] - x[j]
            # grad val_k wrt x_i = v / (2*d)
            # grad val_k wrt x_j = -v / (2*d)
            contrib = w * v / (2.0 * d)
            grad[i] += contrib
            grad[j] -= contrib
            
        # Boundary gradients
        w_bound = weights[len(i_idx):]
        # Sort by bound distance (smallest first, i.e., largest weight)
        bound_order = np.argsort(bound_vals)
        M_b = 50
        for idx in bound_order[:M_b]:
            w = w_bound[idx]
            if w < 1e-6:
                break
            i = idx
            norm = norms[i]
            if norm < 1e-8:
                continue
            # val_i = 1 - |c_i|. grad val_i = -c_i / |c_i|
            contrib = w * (-x[i] / norm)
            grad[i] += contrib
            
        return grad

    # Main loop
    while time.time() - start_time < max_time:
        # Compute current score (exact) for tracking
        current_r = calc_score(x)
        if current_r > best_r:
            best_r = current_r
            best_x = x.copy()
            
        # Compute smooth gradient
        grad = compute_smooth_grad(x, beta)
        
        # Normalize gradient to control step size direction
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-8:
            grad = grad / g_norm
        
        # Adaptive step size
        # Try step
        new_x = x + alpha * grad
        
        # Project to unit circle
        norms = np.linalg.norm(new_x, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            new_x[out_mask] = new_x[out_mask] / norms[out_mask, np.newaxis]
            
        # Evaluate smooth score (faster than exact for acceptance)
        # Actually, let's use exact score for acceptance to be safe, but it's slower.
        # Given N=300, exact score is O(N^2) which is fine.
        new_r = calc_score(new_x)
        
        if new_r > current_r:
            x = new_x
            alpha = min(alpha * 1.05, 0.05)  # Increase step slightly
        else:
            alpha *= 0.8
            if alpha < alpha_min:
                # Random restart / perturbation
                alpha = 0.02
                # Perturb points slightly
                x += np.random.randn(N, 2) * 0.005
                # Project
                norms = np.linalg.norm(x, axis=1)
                out_mask = norms > 1.0
                if np.any(out_mask):
                    x[out_mask] = x[out_mask] / norms[out_mask, np.newaxis]
                # Update best if perturbation helped
                current_r = calc_score(x)
                if current_r > best_r:
                    best_r = current_r
                    best_x = x.copy()
            
        # Check time
        if time.time() - start_time > max_time - 5:
            # Final polish pass with smaller steps if time permits?
            # For now, just break
            break

    # Final check
    final_r = calc_score(best_x)
    if final_r < best_r:
        # Should not happen, but safety
        pass

    return best_x
