"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    # Strategy:
    # 1. Iterate over aspect ratios (W+H=2).
    # 2. For each aspect ratio, use a smooth penalty function approach.
    #    - Objective: Maximize sum(r_i) - Lambda * Penalty
    #    - Penalty: Log-sum-exp of (overlap^2) and (boundary_violation^2)
    #    - This allows gradients to flow smoothly even when circles are touching or near boundaries.
    # 3. Use L-BFGS-B from SciPy for efficient optimization of positions and radius scale.
    #    - Variables: x_1..x_N, y_1..y_N, log_r (scale factor for all radii)
    #    - This reduces the dimensionality of the radius search to 1D per aspect ratio.
    
    from scipy.optimize import minimize, L_BFGS_B
    
    aspect_ratios = [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5]
    
    # Precompute pair indices for efficiency
    pair_indices = []
    for i in range(N):
        for j in range(i+1, N):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    i_idx = pair_indices[:, 0]
    j_idx = pair_indices[:, 1]
    
    for d in aspect_ratios:
        if time.time() > deadline - 5:
            break
            
        W = 1.0 + d
        H = 1.0 - d
        if W <= 0.2 or H <= 0.2:
            continue
            
        # Initial grid
        cols = int(np.sqrt(N * W / H))
        if cols < 1: cols = 1
        if cols > N: cols = N
        rows = int(np.ceil(N / cols))
        
        xs_init = np.zeros(N)
        ys_init = np.zeros(N)
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            xs_init[i] = (c + 0.5) * W / cols
            ys_init[i] = (r_idx + 0.5) * H / rows
            
        # Initial radius estimate
        # Packing density ~0.9 for random, ~0.9069 for hex.
        # Area = W*H. N * pi * r^2 <= 0.9 * W * H
        r_est = np.sqrt(0.9 * W * H / (N * np.pi))
        
        # Variables: [x1..xN, y1..yN, log_r]
        # log_r allows r to be any positive value.
        # We want to maximize sum(r) = N * r.
        # Minimize -N * r + Lambda * Penalty
        
        def objective_and_grad(vars_):
            xs = vars_[:N]
            ys = vars_[N:2*N]
            log_r = vars_[2*N]
            r = np.exp(log_r)
            
            # 1. Penalty for overlaps
            # d_ij = ||p_i - p_j||
            # overlap_ij = max(0, (r_i + r_j) - d_ij)
            # Since all r are equal, r_i + r_j = 2r
            dx = xs[i_idx] - xs[j_idx]
            dy = ys[i_idx] - ys[j_idx]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            
            min_dist = 2 * r
            # Soft penalty: log(sum(exp(overlap^2 / T)))
            # If overlap > 0, this is large.
            # To make it smooth everywhere, we use log-sum-exp of squared violations.
            violations = np.maximum(0, min_dist - dist)
            # Add small constant to avoid log(0) if no violations
            # Actually, if no violations, sum is 0, log is -inf.
            # We want penalty 0 if no violations.
            # Use: log(1 + sum(exp(violations^2 / T)))
            # If violations are 0, exp(0)=1, sum=1, log(2). This is a constant offset.
            # Better: log-sum-exp(violations^2 / T) - log(num_pairs) ? No.
            # Standard smooth max: m + log(sum(exp((x-m)/T)))
            # Here we want sum of penalties.
            # Let's use: log(1 + sum(exp((violations**2) / T)))
            # Gradient w.r.t position:
            # d/dx log(1 + sum(exp(f_j/T))) = sum(exp(f_j/T) * (1/T) * f_j' * 2) / (1 + sum(...))
            
            T = 0.01  # Temperature for smoothing
            exp_terms = np.exp((violations**2) / T)
            sum_exp = np.sum(exp_terms)
            log_sum = np.log(1.0 + sum_exp)
            
            # Penalty for boundaries
            # Violation if x < r or x > W-r
            v_left = np.maximum(0, r - xs)
            v_right = np.maximum(0, xs - (W - r))
            v_bottom = np.maximum(0, r - ys)
            v_top = np.maximum(0, ys - (H - r))
            
            all_boundary_violations = np.concatenate([v_left, v_right, v_bottom, v_top])
            exp_terms_b = np.exp((all_boundary_violations**2) / T)
            sum_exp_b = np.sum(exp_terms_b)
            log_sum_b = np.log(1.0 + sum_exp_b)
            
            total_penalty = log_sum + log_sum_b
            
            # Objective: -N*r + Lambda * total_penalty
            Lambda = 10.0
            val = -N * r + Lambda * total_penalty
            
            # Gradients
            grad = np.zeros_like(vars_)
            
            # Gradient of -N*r w.r.t log_r
            grad[2*N] = -N * r
            
            # Gradient of penalty w.r.t log_r
            # d(2r)/d(log_r) = 2r
            # d(violations)/d(log_r) = 2r for overlaps
            # d(boundary_violations)/d(log_r) = 1 for boundaries (since r changes)
            
            # Overlap gradient
            # d(log(1+sum(exp(v^2/T))))/d(log_r)
            # = (1/(1+sum)) * sum(exp(v^2/T) * (2v/T) * dv/d(log_r))
            # dv/d(log_r) = 2r (for active overlaps, i.e., v>0)
            # Note: v = max(0, 2r - dist). If v>0, dv/d(log_r) = 2r. If v=0, derivative is 0 (subgradient).
            # Using soft max, we effectively weight by exp(v^2/T).
            
            weights = exp_terms / (1.0 + sum_exp)
            # dv/d(log_r) for each pair
            dv_dr = 2 * r
            # Chain rule: d(val)/d(log_r) += Lambda * sum(weights * (2v/T) * dv_dr)
            # Only non-zero if v > 0. The max(0, ...) makes it 0 otherwise.
            # But the exp term handles the smoothness.
            # Actually, if v=0, exp(0)=1, so weight is non-zero.
            # But v=0 means no violation. The derivative of max(0, v) is 0.
            # So we should only include terms where v > 0.
            # However, the log-sum-exp is smooth.
            # Let's compute the gradient properly.
            # Let S = 1 + sum(exp(v_k^2/T))
            # d/d(log_r) log(S) = (1/S) * sum(exp(v_k^2/T) * (2 v_k / T) * dv_k/d(log_r))
            # dv_k/d(log_r) = 2r if v_k > 0 else 0
            
            active = violations > 1e-9
            if np.any(active):
                d_v_dr = np.zeros_like(violations)
                d_v_dr[active] = 2 * r
                grad_overlap_r = np.sum(weights[active] * (2 * violations[active] / T) * d_v_dr[active])
                grad[2*N] += Lambda * grad_overlap_r
            
            # Gradient of overlap penalty w.r.t positions
            # d(v_k)/dx_i = - (dx_k / dist_k) if active
            # d(v_k)/dx_j = (dx_k / dist_k) if active
            # grad_x_i += Lambda * sum_k (weights_k * (2 v_k / T) * d(v_k)/dx_i)
            
            if np.any(active):
                # We need to scatter the gradients
                # weight_k = weights[active] * (2 * violations[active] / T)
                w_active = weights[active] * (2 * violations[active] / T)
                
                # d(v_k)/dx_i = - dx_k / dist_k
                # d(v_k)/dx_j = dx_k / dist_k
                
                # We have dx, dy for all pairs.
                # dist is computed.
                # We need to map back to indices.
                
                # Create temporary arrays for gradients
                grad_x_temp = np.zeros(N)
                grad_y_temp = np.zeros(N)
                
                # For each active pair k:
                # i = i_idx[k], j = j_idx[k]
                # factor = w_active[k]
                # d_v_dx_i = - dx[k] / dist[k]
                # d_v_dx_j = dx[k] / dist[k]
                
                # Vectorized scatter-add
                # We can use np.add.at
                active_pairs = pair_indices[active]
                a_i = active_pairs[:, 0]
                a_j = active_pairs[:, 1]
                
                dx_active = dx[active]
                dy_active = dy[active]
                dist_active = dist[active]
                
                # Normalized directions
                nx = dx_active / dist_active
                ny = dy_active / dist_active
                
                # Gradients
                # d(val)/dx_i = - Lambda * sum w * nx
                # d(val)/dx_j = Lambda * sum w * nx
                
                np.add.at(grad_x_temp, a_i, -Lambda * w_active * nx)
                np.add.at(grad_x_temp, a_j, Lambda * w_active * nx)
                
                np.add.at(grad_y_temp, a_i, -Lambda * w_active * ny)
                np.add.at(grad_y_temp, a_j, Lambda * w_active * ny)
                
                grad[:N] += grad_x_temp
                grad[N:2*N] += grad_y_temp
            
            # Boundary penalty gradient
            # Similar logic.
            # v_left = max(0, r - x)
            # d(v_left)/dx = -1 if active
            # d(v_left)/d(log_r) = 1 if active
            
            # Combine all boundary violations into one array for weight calculation
            # But we need to map back to specific variables.
            
            # Let's do it separately for clarity and efficiency
            # Left: v = max(0, r - x)
            active_left = v_left > 1e-9
            if np.any(active_left):
                exp_l = np.exp((v_left[active_left]**2) / T)
                sum_l = np.sum(exp_l)
                log_l = np.log(1.0 + sum_l)
                weights_l = exp_l / (1.0 + sum_l)
                
                # d(v)/d(log_r) = 1
                grad[2*N] += Lambda * np.sum(weights_l * (2 * v_left[active_left] / T) * 1.0)
                
                # d(v)/dx = -1
                grad_x_l = np.zeros(N)
                np.add.at(grad_x_l, np.where(active_left)[0], -Lambda * weights_l * (2 * v_left[active_left] / T) * (-1.0))
                grad[:N] += grad_x_l
            
            # Right: v = max(0, x - (W-r))
            active_right = v_right > 1e-9
            if np.any(active_right):
                exp_r = np.exp((v_right[active_right]**2) / T)
                sum_r = np.sum(exp_r)
                weights_r = exp_r / (1.0 + sum_r)
                
                # d(v)/d(log_r) = 1 (since W-r decreases as r increases? No, W-r is lower bound. x > W-r. v = x - W + r. dv/dr = 1)
                grad[2*N] += Lambda * np.sum(weights_r * (2 * v_right[active_right] / T) * 1.0)
                
                # d(v)/dx = 1
                grad_x_r = np.zeros(N)
                np.add.at(grad_x_r, np.where(active_right)[0], Lambda * weights_r * (2 * v_right[active_right] / T) * 1.0)
                grad[:N] += grad_x_r
            
            # Bottom: v = max(0, r - y)
            active_bottom = v_bottom > 1e-9
            if np.any(active_bottom):
                exp_b = np.exp((v_bottom[active_bottom]**2) / T)
                sum_b = np.sum(exp_b)
                weights_b = exp_b / (1.0 + sum_b)
                
                grad[2*N] += Lambda * np.sum(weights_b * (2 * v_bottom[active_bottom] / T) * 1.0)
                
                grad_y_b = np.zeros(N)
                np.add.at(grad_y_b, np.where(active_bottom)[0], -Lambda * weights_b * (2 * v_bottom[active_bottom] / T) * (-1.0))
                grad[N:2*N] += grad_y_b
            
            # Top: v = max(0, y - (H-r))
            active_top = v_top > 1e-9
            if np.any(active_top):
                exp_t = np.exp((v_top[active_top]**2) / T)
                sum_t = np.sum(exp_t)
                weights_t = exp_t / (1.0 + sum_t)
                
                grad[2*N] += Lambda * np.sum(weights_t * (2 * v_top[active_top] / T) * 1.0)
                
                grad_y_t = np.zeros(N)
                np.add.at(grad_y_t, np.where(active_top)[0], Lambda * weights_t * (2 * v_top[active_top] / T) * 1.0)
                grad[N:2*N] += grad_y_t
                
            return val, grad
        
        # Initial guess
        # Start with small radius to avoid huge initial penalties
        r_init = 0.5 * r_est
        log_r_init = np.log(r_init)
        
        x0 = np.concatenate([xs_init, ys_init, [log_r_init]])
        
        # Bounds for L-BFGS-B
        # x in [0, W], y in [0, H]
        # log_r in [log(0.01), log(1.0)]
        bounds = []
        for i in range(N):
            bounds.append((0.0, W))
            bounds.append((0.0, H))
        bounds.append((np.log(0.01), np.log(1.0)))
        
        # Optimize
        try:
            result = minimize(
                lambda v: objective_and_grad(v)[0],
                x0,
                jac=lambda v: objective_and_grad(v)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 50}
            )
            
            if result.success or result.fun < 0: # If it found a minimum
                xs_opt = result.x[:N]
                ys_opt = result.x[N:2*N]
                log_r_opt = result.x[2*N]
                r_opt = np.exp(log_r_opt)
                
                # Check validity
                valid = True
                for i in range(N):
                    if xs_opt[i] < r_opt - 1e-5 or xs_opt[i] > W - r_opt + 1e-5:
                        valid = False
                    if ys_opt[i] < r_opt - 1e-5 or ys_opt[i] > H - r_opt + 1e-5:
                        valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        dx = xs_opt[i] - xs_opt[j]
                        dy = ys_opt[i] - ys_opt[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = 2 * r_opt
                        if dist_sq < min_dist*min_dist - 1e-5:
                            valid = False
                
                if valid:
                    total_r = N * r_opt
                    if total_r > best_sum_r:
                        best_sum_r = total_r
                        best_config = np.column_stack((xs_opt, ys_opt, np.full(N, r_opt)))
                        best_W = W
                        best_H = H
                        
        except Exception as e:
            # Fallback to grid if optimizer fails
            pass

    # Fallback if no valid config found
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Verify best_config one last time
    xs = best_config[:, 0]
    ys = best_config[:, 1]
    rs = best_config[:, 2]
    
    valid = True
    for i in range(N):
        if xs[i] < rs[i] - 1e-5 or xs[i] > best_W - rs[i] + 1e-5:
            valid = False
        if ys[i] < rs[i] - 1e-5 or ys[i] > best_H - rs[i] + 1e-5:
            valid = False
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist*min_dist - 1e-5:
                valid = False
                
    if valid:
        return best_config
    else:
        # If invalid due to numerical errors, return safe grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
