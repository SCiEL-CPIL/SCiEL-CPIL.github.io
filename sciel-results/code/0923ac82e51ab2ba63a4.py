import numpy as np
import time

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # Step width
    dw = 2.0 / N
    
    # Overlap at shift k (in steps) for step function h:
    # overlap_k = dw * sum_i min(h[i], 1 - h[(i+k) % N])
    # For circular shifts (periodic extension on [0,2])
    
    def compute_overlaps(h):
        """Compute overlap at all N circular shifts."""
        overlaps = np.zeros(N)
        for k in range(N):
            # h_shifted[i] = h[(i+k) % N]
            h_shifted = np.roll(h, -k)
            # min(h[i], 1 - h_shifted[i])
            overlaps[k] = np.sum(np.minimum(h, 1.0 - h_shifted)) * dw
        return overlaps
    
    def soft_max(overlaps, beta):
        """Soft max surrogate for max(overlaps)."""
        m = np.max(overlaps)
        return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
    
    def soft_max_grad(overlaps, beta):
        """Gradient of soft max w.r.t. overlaps."""
        m = np.max(overlaps)
        exps = np.exp(beta * (overlaps - m))
        weights = exps / np.sum(exps)
        return weights
    
    # Start from a non-trivial initial guess
    # Try a binary-like pattern: alternating blocks
    h = np.zeros(N)
    # Initialize with a pattern that has low self-correlation
    # Use a "balanced" binary sequence approach
    h[:] = 0.5
    
    # Try different initial patterns
    best_h = h.copy()
    best_score = np.inf
    
    patterns = [
        np.full(N, 0.5),
        np.tile([1.0, 0.0], N // 2),
        np.tile([1.0, 0.0, 0.0, 1.0], N // 4),
        np.sin(np.linspace(0, 4 * np.pi, N)) * 0.5 + 0.5,
        np.sin(np.linspace(0, 6 * np.pi, N)) * 0.5 + 0.5,
    ]
    
    for p in patterns:
        if len(p) != N:
            continue
        p = p.copy()
        # Ensure sum = N/2 (integral = 1)
        p = np.clip(p, 0, 1)
        # Adjust to have correct sum
        target_sum = N / 2.0
        current_sum = np.sum(p)
        if current_sum > 0:
            p = p * (target_sum / current_sum)
            p = np.clip(p, 0, 1)
            # Re-adjust if clipping changed sum
            diff = target_sum - np.sum(p)
            if abs(diff) > 1e-10:
                # Distribute difference
                mask = (p > 0) & (p < 1)
                if np.any(mask):
                    p[mask] += diff / np.sum(mask)
                    p = np.clip(p, 0, 1)
        
        overlaps = compute_overlaps(p)
        score = np.max(overlaps)
        if score < best_score:
            best_score = score
            best_h = p.copy()
    
    # Gradient-based optimization using soft-max
    h = best_h.copy()
    
    # Use projected gradient descent
    lr = 0.01
    beta_start = 10.0
    beta_end = 100.0
    
    num_iters = 500
    for it in range(num_iters):
        if time.time() > DEADLINE:
            break
        
        # Animate beta
        t = it / max(num_iters - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        overlaps = compute_overlaps(h)
        score = soft_max(overlaps, beta)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Gradient of soft_max w.r.t. h
        weights = soft_max_grad(overlaps, beta)
        
        # Gradient of overlaps w.r.t. h
        # overlap_k = dw * sum_i min(h[i], 1 - h[(i+k)%N])
        # d(overlap_k)/d(h[i]) = dw * (1 if h[i] < 1 - h[(i+k)%N] else 0) for the min term
        # But min is not differentiable; use smooth approximation
        # d/dx min(x, y) ≈ x/(x+y) * (y-x)/(x+y) ... use soft-min
        # Simpler: use the fact that for the subgradient, we can use a smoothed version
        
        # Use a smooth approximation: min(a,b) ≈ -1/beta_sm * log(exp(-beta_sm*a) + exp(-beta_sm*b))
        beta_sm = 50.0
        
        grad_h = np.zeros(N)
        for k in range(N):
            h_shifted = np.roll(h, -k)
            # soft min of h[i] and 1-h_shifted[i]
            a = h
            b = 1.0 - h_shifted
            # Gradient of soft-min w.r.t. a: exp(-beta*a) / (exp(-beta*a) + exp(-beta*b))
            exp_a = np.exp(-beta_sm * a)
            exp_b = np.exp(-beta_sm * b)
            denom = exp_a + exp_b
            grad_min_wrt_a = exp_a / denom  # d(softmin)/d(a)
            
            # Also need gradient w.r.t. b: d(softmin)/d(b) = exp_b / denom
            # And b = 1 - h_shifted, so d(b)/d(h[j]) = -1 if j = (i+k)%N
            
            grad_k = weights[k] * dw * grad_min_wrt_a
            grad_h += grad_k
            
            # Gradient from the b term
            grad_min_wrt_b = exp_b / denom
            # d(overlap_k)/d(h[j]) for j = (i+k)%N: -grad_min_wrt_b
            # So we add to grad_h[(i+k)%N] the term: -weights[k]*dw*grad_min_wrt_b[i]
            grad_from_b = -weights[k] * dw * grad_min_wrt_b
            # This contributes to h[(i+k)%N]
            # grad_h[(i+k)%N] += grad_from_b[i]
            # Equivalent to: grad_h += roll(grad_from_b, k)
            grad_h += np.roll(grad_from_b, k)
        
        # Projected gradient step
        h_new = h - lr * grad_h
        h_new = np.clip(h_new, 0, 1)
        
        # Enforce sum constraint: sum(h) = N/2
        target_sum = N / 2.0
        current_sum = np.sum(h_new)
        if abs(current_sum - target_sum) > 1e-10:
            diff = target_sum - current_sum
            # Adjust by adding diff/N to all, then clip and re-adjust
            h_new = h_new + diff / N
            h_new = np.clip(h_new, 0, 1)
            # Final adjustment
            diff2 = target_sum - np.sum(h_new)
            if abs(diff2) > 1e-10:
                mask = (h_new > 0) & (h_new < 1)
                if np.any(mask):
                    h_new[mask] += diff2 / np.sum(mask)
                    h_new = np.clip(h_new, 0, 1)
        
        h = h_new
    
    # Final polish: ensure exact sum
    h = best_h.copy()
    target_sum = N / 2.0
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) > 1e-10:
        diff = target_sum - current_sum
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    
    # Verify
    final_overlaps = compute_overlaps(h)
    final_score = np.max(final_overlaps)
    
    if final_score < best_score:
        best_h = h.copy()
        best_score = final_score
    
    return best_h
