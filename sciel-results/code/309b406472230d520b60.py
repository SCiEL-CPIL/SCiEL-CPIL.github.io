import numpy as np
import time
import scipy.optimize as opt

N = 300

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    best_score = 0.0
    
    # Precompute indices for pairwise distances
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def _score(pts):
        """Compute the packing score."""
        d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(d**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists[i_idx, j_idx]) / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_min = np.min(1.0 - rad)
        return min(pair_min, bound_min)
    
    def _soft_min_weights(vals, k_soft):
        """Compute soft-min weights with numerical stability."""
        current_min = np.min(vals)
        shifted = vals - current_min
        threshold = 50.0 / k_soft
        mask = shifted < threshold
        if np.any(mask):
            close_shifted = shifted[mask]
            max_close = np.max(close_shifted)
            if max_close > 0:
                exp_terms = np.exp(-k_soft * (close_shifted - max_close))
            else:
                exp_terms = np.exp(-k_soft * close_shifted)
            
            sum_exp = np.sum(exp_terms)
            if sum_exp > 0:
                weights = np.zeros_like(vals)
                weights[mask] = exp_terms / sum_exp
            else:
                weights = np.ones_like(vals) / len(vals)
        else:
            weights = np.ones_like(vals) / len(vals)
        return weights, current_min
    
    # Optimization parameters
    alpha = 0.02
    cooling = 0.99995
    k_soft_start = 3.0
    k_soft_end = 50.0 # Lower end for phase 1, L-BFGS will use higher
    max_iter = 8000
    iter_count = 0
    stall_count = 0
    stall_threshold = 150
    k_soft = k_soft_start
    
    # Phase 1: Main optimization (Gradient Ascent)
    # Run for about 40% of the time limit
    phase1_time_limit = time_limit * 0.4
    
    while time.time() - start_time < phase1_time_limit and iter_count < max_iter:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        weights, _ = _soft_min_weights(all_vals, k_soft)
        
        grads = np.zeros_like(points)
        
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        points += alpha * grads
        
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        alpha *= cooling
        
        progress = iter_count / max_iter
        k_soft = k_soft_start + (k_soft_end - k_soft_start) * (progress ** 1.5)
        
        if stall_count > stall_threshold:
            noise_scale = alpha * 2.0
            points += np.random.normal(0, noise_scale, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
            stall_count = 0
            alpha = max(alpha, 0.005)
        
        iter_count += 1

    # Phase 2: L-BFGS-B Polish
    # Use the best points found so far
    points = best_points.copy()
    
    def _lbfgsb_obj_and_grad(x_flat, k_val):
        pts = x_flat.reshape(-1, 2)
        
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        # Soft-min calculation
        # m_k(v) = -1/k * ln(sum(exp(-k*v)))
        # To minimize -m_k(v), we minimize 1/k * ln(sum(exp(-k*v)))
        
        # Numerical stability for log-sum-exp
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        # We only care about values close to the min for the gradient, 
        # but for the objective value, we need the sum.
        # However, exp(-k * shifted) will be 0 for large shifted values.
        
        exp_terms = np.exp(-k_val * shifted)
        sum_exp = np.sum(exp_terms)
        
        # Objective value: -m_k(v) = 1/k * ln(sum_exp) + min_val
        # Wait, m_k(v) = -1/k * ln(sum(exp(-k*v)))
        # sum(exp(-k*v)) = exp(-k*min_val) * sum(exp(-k*(v-min_val)))
        # ln(sum(exp(-k*v))) = -k*min_val + ln(sum(exp(-k*(v-min_val))))
        # m_k(v) = -1/k * (-k*min_val + ln(sum_exp_shifted)) = min_val - 1/k * ln(sum_exp_shifted)
        # We want to maximize m_k(v), so minimize -m_k(v) = -min_val + 1/k * ln(sum_exp_shifted)
        
        obj_val = -min_val + (1.0 / k_val) * np.log(sum_exp)
        
        # Gradient
        # grad(-m_k) = - grad(m_k) = - sum(w_j * grad(v_j))
        # w_j = exp(-k*v_j) / sum(exp(-k*v_l)) = exp(-k*(v_j-min_val)) / sum_exp
        weights = exp_terms / sum_exp
        
        grads = np.zeros_like(pts)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        # Correction: use pts not points
        bound_grads = - (pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # The gradient of the objective function to minimize (-m_k) is -grad(m_k)
        # grad(m_k) = sum(w_j * grad(v_j))
        # So we return -grads
        return obj_val, -grads.flatten()

    # Run L-BFGS-B
    # We can try to increase k_val if we have time, but a fixed high k is usually good for polish
    k_lbfgsb = 100.0
    
    x0 = points.flatten()
    
    try:
        # Use maxiter based on time? L-BFGS-B doesn't support time limit directly easily.
        # We can set a high maxiter and hope it converges or stalls.
        # Or we can run it in a loop with decreasing time slices? 
        # Standard practice: just give it a lot of iterations and let it run until time runs out?
        # scipy.optimize.minimize doesn't have a time limit callback.
        # However, we can check time inside the callback? No, callback is for plotting/logging.
        
        # Alternative: Run L-BFGS-B with a reasonable maxiter (e.g., 200) and see if it finishes.
        # If it finishes quickly, we can restart with higher k or just return.
        # Given the 150s limit, and Phase 1 taking ~40s, we have ~70s left.
        # L-BFGS-B is fast. 200-500 iterations should be well within 70s for N=300.
        
        result = opt.minimize(
            fun=_lbfgsb_obj_and_grad, 
            x0=x0, 
            args=(k_lbfgsb,), 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < _lbfgsb_obj_and_grad(x0, k_lbfgsb)[0]:
            new_points = result.x.reshape(-1, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
    except Exception as e:
        print(f"L-BFGS-B failed: {e}")
        
    # Final check and return
    return best_points
