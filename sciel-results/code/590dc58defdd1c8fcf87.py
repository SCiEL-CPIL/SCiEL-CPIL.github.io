"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _overlap_max(h, shift):
    """Compute max overlap of h with h shifted by 'shift' positions, cyclically.
    overlap(k) = sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize max over k of overlap(k).
    """
    N_local = len(h)
    h = np.asarray(h, dtype=np.float64)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(N_local):
        shifted = np.roll(one_minus_h, k)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _overlap_grad(h, shift):
    """Gradient of overlap(h, shift) w.r.t. h.
    overlap = sum_i h[i] * (1 - h[(i+shift)%N])
    d/dh[j] = (1 - h[(j+shift)%N]) - h[(j-shift)%N]
    """
    N_local = len(h)
    h = np.asarray(h, dtype=np.float64)
    grad = np.zeros(N_local)
    one_minus_h = 1.0 - h
    # term 1: for each i, derivative w.r.t. h[i] of h[i]*(1-h[i+k]) is (1-h[i+k])
    # term 2: for each i, derivative w.r.t. h[i] of h[i]*(1-h[i+k]) includes -h[i-k]*(1) from the other factor
    # Actually let's be careful:
    # overlap = sum_i h[i] * (1 - h[(i+k)%N])
    # Let j = i, so term is h[j] * (1 - h[(j+k)%N])
    # d/dh[j]: (1 - h[(j+k)%N]) + sum over i where (i+k)%N == j, i.e., i = (j-k)%N: h[(j-k)%N] * (-1)
    # So d/dh[j] = (1 - h[(j+k)%N]) - h[(j-k)%N]
    for j in range(N_local):
        grad[j] = (1.0 - h[(j + shift) % N_local]) - h[(j - shift) % N_local]
    return grad


def _smooth_max(overlaps, beta):
    """Log-sum-exp style smooth max."""
    max_val = np.max(overlaps)
    if beta < 1e-10:
        return max_val
    # soft max: max + (1/beta) * log(sum(exp(beta*(x - max))))
    exp_vals = np.exp(beta * (overlaps - max_val))
    return max_val + np.log(np.sum(exp_vals)) / beta


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with 0.5 everywhere (sum = N/2)
    h = np.full(N, 0.5)
    
    # We'll use projected gradient descent on the smooth max of overlaps
    # Constraint: sum(h) = N/2, h in [0,1]
    
    best_h = h.copy()
    best_val = _overlap_max(h, 0)  # This is just the max over all shifts
    
    # Actually, let me compute the true max overlap properly
    def true_max_overlap(h_vec):
        N_loc = len(h_vec)
        max_ov = 0.0
        for k in range(N_loc):
            shifted = np.roll(1.0 - h_vec, k)
            ov = np.dot(h_vec, shifted)
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    best_val = true_max_overlap(h)
    
    # Gradient of the smooth max of overlaps
    def smooth_max_and_grad(h_vec, beta):
        N_loc = len(h_vec)
        overlaps = np.zeros(N_loc)
        for k in range(N_loc):
            shifted = np.roll(1.0 - h_vec, k)
            overlaps[k] = np.dot(h_vec, shifted)
        
        max_ov = np.max(overlaps)
        exp_vals = np.exp(beta * (overlaps - max_ov))
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of smooth max = sum_k weights[k] * grad_overlap_k
        grad = np.zeros(N_loc)
        for k in range(N_loc):
            if weights[k] > 1e-15:
                grad += weights[k] * _overlap_grad(h_vec, k)
        
        return max_ov + np.log(np.sum(exp_vals)) / beta, grad, overlaps
    
    # Projected gradient descent
    # Projection: given a vector v, project onto {h in [0,1]^N : sum(h) = N/2}
    def project(v):
        # First clip to [0,1]
        v = np.clip(v, 0.0, 1.0)
        # Now adjust to satisfy sum constraint
        # We need sum(v) = N/2
        # Use bisection on the Lagrange multiplier
        target_sum = N / 2.0
        
        def sum_at_tau(tau):
            h_t = np.clip(v - tau, 0.0, 1.0)
            return np.sum(h_t)
        
        # Find tau such that sum_at_tau(tau) = target_sum
        # sum_at_tau is decreasing in tau
        lo, hi = -2.0, 2.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            s = sum_at_tau(mid)
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        tau = (lo + hi) / 2.0
        h_proj = np.clip(v - tau, 0.0, 1.0)
        return h_proj
    
    # Adaptive learning rate with backtracking line search
    beta_start = 5.0
    beta_end = 500.0
    lr = 0.01
    
    num_iterations = 5000
    
    for it in range(num_iterations):
        if time.time() > deadline:
            break
        
        # Progressively increase beta
        progress = it / num_iterations
        beta = beta_start * (beta_end / beta_start) ** progress
        
        val, grad, overlaps = smooth_max_and_grad(h, beta)
        
        # Projected gradient step
        # h_new = project(h - lr * grad)
        h_trial = h - lr * grad
        h_proj = project(h_trial)
        
        # Check if this improves the true objective
        new_true_val = true_max_overlap(h_proj)
        
        if new_true_val < best_val:
            best_val = new_true_val
            best_h = h_proj.copy()
        
        # Backtracking line search if needed
        if new_true_val >= val - 1e-12:  # Not making progress on smooth objective
            # Try smaller step
            lr_new = lr * 0.5
            h_trial2 = h - lr_new * grad
            h_proj2 = project(h_trial2)
            new_true_val2 = true_max_overlap(h_proj2)
            if new_true_val2 < new_true_val:
                h = h_proj2
                lr = lr_new
                if new_true_val2 < best_val:
                    best_val = new_true_val2
                    best_h = h.copy()
            else:
                # Accept anyway to avoid stalling, but reduce lr
                h = h_proj
                lr = max(lr * 0.5, 1e-6)
        else:
            h = h_proj
            # Increase lr slightly if making good progress
            if new_true_val < best_val:
                lr = min(lr * 1.1, 0.1)
    
    # Polish phase with high beta and tiny LR
    if time.time() < deadline - 5.0:
        h = best_h.copy()
        lr_polish = 1e-4
        beta_polish = 2000.0
        num_polish = 2000
        
        for it in range(num_polish):
            if time.time() > deadline:
                break
            
            val, grad, overlaps = smooth_max_and_grad(h, beta_polish)
            h_trial = h - lr_polish * grad
            h_proj = project(h_trial)
            new_true_val = true_max_overlap(h_proj)
            
            if new_true_val < best_val:
                best_val = new_true_val
                best_h = h_proj.copy()
                h = h_proj
            else:
                h = h_proj
                lr_polish = max(lr_polish * 0.8, 1e-7)
    
    # Final projection to ensure exact constraints
    h_final = project(best_h)
    # Ensure sum is exactly N/2
    diff = N / 2.0 - np.sum(h_final)
    if abs(diff) > 1e-10:
        # Distribute the difference evenly, respecting bounds
        h_final = h_final + diff / N
        h_final = np.clip(h_final, 0.0, 1.0)
        # Re-project to fix sum again
        h_final = project(h_final)
    
    return h_final

# EVOLVE-BLOCK-END
