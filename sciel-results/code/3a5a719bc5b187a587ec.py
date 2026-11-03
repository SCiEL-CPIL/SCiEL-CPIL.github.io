"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # We need to find h of length N with:
    # - h[i] in [0, 1]
    # - sum(h) = N/2
    # - Minimize max over shifts k of overlap(h, shift_k(1-h))
    
    # The overlap for shift k is: sum_i h[i] * (1 - h[(i+k) % N])
    # We want to minimize the maximum of these overlaps.
    
    # Let's think about this differently. If h is constant 0.5, then:
    # overlap(k) = sum_i 0.5 * 0.5 = N * 0.25 = N/4
    # So the baseline score is N/4 = 25.
    
    # The score reported is the upper bound on C5, which seems to be the max overlap.
    # Wait, let me re-read: "score is the upper bound on C5" and "evaluator measures the raw value (lower is better)"
    # And "search score is seed/value" where seed program gives 1.0.
    
    # The seed returns np.full(N, 0.5), so sum = N/2 = 50. Good.
    # For h = 0.5 everywhere, overlap(k) = sum_i 0.5*(1-0.5) = N*0.25 = 25.
    # So seed value = 25, and search score = seed/value = 25/25 = 1.0.
    
    # We need to minimize max_k overlap(k) where overlap(k) = sum_i h[i]*(1-h[(i+k)%N])
    
    # Let's use a gradient-based approach with projection onto the constraint set.
    # Constraints: 0 <= h[i] <= 1, sum(h) = N/2
    
    # Objective: f(h) = max_k overlap(k)
    # This is a nonsmooth objective (max of linear functions in h for fixed shifts).
    
    # Let's use subgradient or a smooth approximation.
    
    # Actually, let's think about what structure minimizes this.
    # If h alternates between 0 and 1 in some pattern, we might reduce overlaps.
    
    # Let's try a simple approach: use scipy's minimize with a smooth surrogate.
    
    from scipy.optimize import minimize
    
    def compute_overlaps(h):
        """Compute all circular shift overlaps."""
        h = np.asarray(h)
        n = len(h)
        overlaps = np.zeros(n)
        for k in range(n):
            shifted = np.roll(h, k)
            overlaps[k] = np.sum(h * (1 - shifted))
        return overlaps
    
    def objective_and_grad(h):
        """Smooth approximation of max overlap using log-sum-exp."""
        overlaps = compute_overlaps(h)
        beta = 10.0
        # Log-sum-exp for numerical stability
        max_overlap = np.max(overlaps)
        if max_overlap < 1e-10:
            return max_overlap, np.zeros_like(h)
        
        # f = max_overlap + (1/beta) * log(sum(exp(beta * (overlaps - max_overlap))))
        # Gradient: f = max_overlap + (1/beta) * log(sum_j exp(beta*(o_j - o_max)))
        # df/dh_i = (1/beta) * sum_j beta * exp(beta*(o_j - o_max)) / sum_j exp(beta*(o_j - o_max)) * do_j/dh_i
        # do_j/dh_i = (1 - h[(i+j)%N])
        
        exp_terms = np.exp(beta * (overlaps - max_overlap))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp  # softmax-like weights
        
        # f = max_overlap + (1/beta) * log(sum_exp)
        f_val = max_overlap + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient: sum_j weights[j] * do_j/dh_i
        # do_j/dh_i = (1 - h[(i+j) % N])
        grad = np.zeros_like(h)
        for i in range(len(h)):
            for j in range(len(h)):
                grad[i] += weights[j] * (1 - h[(i + j) % len(h)])
        
        return f_val, grad
    
    # Initialize with a non-trivial pattern
    # Try alternating high/low values
    h0 = np.zeros(N)
    for i in range(N):
        if i % 2 == 0:
            h0[i] = 0.7
        else:
            h0[i] = 0.3
    # Adjust to ensure sum = N/2
    h0 = h0 - np.mean(h0) + N / 2.0 / N
    h0 = np.clip(h0, 0, 1)
    # Re-adjust sum
    diff = N / 2.0 - np.sum(h0)
    h0 = h0 + diff / N
    h0 = np.clip(h0, 0, 1)
    # Final adjustment
    diff = N / 2.0 - np.sum(h0)
    if abs(diff) > 1e-6:
        # Distribute evenly
        h0 = h0 + diff / N
        h0 = np.clip(h0, 0, 1)
    
    # Use Nesterov accelerated gradient with projection
    def project(h):
        """Project onto {0 <= h[i] <= 1, sum(h) = N/2}"""
        h = np.clip(h, 0, 1)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-8:
            # Simple adjustment: add diff/N, then re-clip
            h = h + diff / N
            h = np.clip(h, 0, 1)
            # If still not exact, do a more careful projection
            # This is a convex problem, can be solved via bisection on Lagrange multiplier
            lo, hi = -1.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2
                h_proj = np.clip(h - mid, 0, 1)
                s = np.sum(h_proj)
                if s > N / 2.0:
                    lo = mid
                else:
                    hi = mid
            h = np.clip(h - (lo + hi) / 2, 0, 1)
        return h
    
    # Nesterov accelerated gradient descent
    h = h0.copy()
    y = h0.copy()
    t = 1.0
    lr = 0.01
    best_h = h.copy()
    best_val = float('inf')
    
    # Compute initial value
    overlaps = compute_overlaps(h)
    best_val = np.max(overlaps)
    
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute objective and gradient at y
        f_val, grad = objective_and_grad(y)
        
        # Nesterov update
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        alpha = (t - 1) / t_new
        
        y_new = project(y - lr * grad)
        h_new = y_new + alpha * (y_new - h)
        h_new = project(h_new)
        
        # Evaluate
        overlaps = compute_overlaps(h_new)
        val = np.max(overlaps)
        
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
        
        h = h_new
        y = y_new
        t = t_new
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr = lr * 0.95
    
    # Final polish with more precise method
    # Try a few random restarts if time permits
    if time.time() < deadline - 20:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            h_r = np.random.uniform(0, 1, N)
            h_r = h_r - np.mean(h_r) + N / 2.0 / N
            h_r = np.clip(h_r, 0, 1)
            diff = N / 2.0 - np.sum(h_r)
            h_r = h_r + diff / N
            h_r = np.clip(h_r, 0, 1)
            
            # Quick optimization
            h_r = h_r.copy()
            y_r = h_r.copy()
            t_r = 1.0
            lr_r = 0.01
            
            for it in range(1000):
                if time.time() > deadline - 5:
                    break
                
                f_val, grad = objective_and_grad(y_r)
                t_new = (1 + np.sqrt(1 + 4 * t_r * t_r)) / 2
                alpha = (t_r - 1) / t_new
                
                y_new = project(y_r - lr_r * grad)
                h_new = y_new + alpha * (y_new - h_r)
                h_new = project(h_new)
                
                h_r = h_new
                y_r = y_new
                t_r = t_new
                
                if it % 50 == 0:
                    lr_r = lr_r * 0.95
            
            overlaps = compute_overlaps(h_r)
            val = np.max(overlaps)
            if val < best_val:
                best_val = val
                best_h = h_r.copy()
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-10:
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0, 1)
        # More careful projection
        lo, hi = -1.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            h_proj = np.clip(best_h - mid, 0, 1)
            s = np.sum(h_proj)
            if s > N / 2.0:
                lo = mid
            else:
                hi = mid
        best_h = np.clip(best_h - (lo + hi) / 2, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
