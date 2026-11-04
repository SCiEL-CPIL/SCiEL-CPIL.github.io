import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    """
    Returns the optimal sequence of step heights.
    
    We minimize the maximum overlap O_k = sum_i h_i * (1 - h_{(i+k)%N}) over k.
    This is equivalent to maximizing the minimum circular autocorrelation C_k.
    
    Strategy:
    1. Analytical lower bound: uniform h=0.5 gives O_k = N/4 = 25 for all k.
    2. Try to beat this with structured patterns (e.g., alternating, block patterns).
    3. Use gradient-based optimization to refine a good initial guess.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin of 5s
    
    def elapsed():
        return time.time() - start_time
    
    def overlap_max(h):
        """Compute max overlap over all shifts."""
        h = np.clip(h, 0, 1)
        # Normalize to sum exactly N/2
        s = h.sum()
        if s > 0:
            h = h * (N / 2.0) / s
        h = np.clip(h, 0, 1)
        # Re-normalize after clipping
        s2 = h.sum()
        if s2 > 0:
            h = h * (N / 2.0) / s2
        
        # Compute overlaps for all shifts
        max_ov = 0.0
        for k in range(1, N):
            shifted = np.roll(h, k)
            ov = np.sum(h * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov, h
    
    def objective(x):
        """Objective for optimizer: max overlap."""
        h = x
        max_ov, _ = overlap_max(h)
        return max_ov
    
    # Start with uniform
    best_h = np.full(N, 0.5)
    best_score = overlap_max(best_h)[0]  # Should be 25.0
    
    # Try some structured initializations
    inits = []
    
    # Alternating
    alt = np.array([0.0, 1.0] * (N // 2))
    inits.append(alt)
    
    # Block patterns: various block sizes
    for block in [2, 4, 5, 10, 20, 25]:
        nblocks = N // block
        pattern = np.zeros(N)
        for i in range(nblocks):
            start = i * block
            pattern[start:start+block//2] = 0.0
            pattern[start+block//2:start+block] = 1.0
        inits.append(pattern)
    
    # Random patterns with sum constraint
    np.random.seed(42)
    for _ in range(10):
        r = np.random.rand(N)
        # Scale to have mean 0.5
        r = (r - r.mean()) / (r.std() + 1e-10) * 0.25 + 0.5
        r = np.clip(r, 0, 1)
        # Adjust sum
        s = r.sum()
        if s > 0:
            r = r * (N/2.0) / s
        r = np.clip(r, 0, 1)
        inits.append(r)
    
    # Evaluate initializations
    for h_init in inits:
        if elapsed() > deadline - 30:
            break
        score, h_norm = overlap_max(h_init)
        if score < best_score:
            best_score = score
            best_h = h_norm.copy()
    
    # Gradient-based optimization from best initialization
    if elapsed() < deadline - 10:
        # Use L-BFGS-B with bounds
        x0 = best_h.copy()
        
        def obj_grad(x):
            h = np.clip(x, 0, 1)
            # Normalize
            s = h.sum()
            if s > 0:
                h = h * (N/2.0) / s
            h = np.clip(h, 0, 1)
            s2 = h.sum()
            if s2 > 0:
                h = h * (N/2.0) / s2
            
            # Compute max overlap and its subgradient
            max_ov = 0.0
            argmax_k = 0
            for k in range(1, N):
                shifted = np.roll(h, k)
                ov = np.sum(h * (1.0 - shifted))
                if ov > max_ov:
                    max_ov = ov
                    argmax_k = k
            
            # Gradient of O_k w.r.t. h_i
            # O_k = sum_i h_i (1 - h_{i+k})
            # dO_k/dh_i = (1 - h_{i+k}) - h_{i-k}
            shifted_plus = np.roll(h, argmax_k)
            shifted_minus = np.roll(h, -argmax_k)
            grad = (1.0 - shifted_plus) - shifted_minus
            
            # Project gradient back through normalization (approximate)
            # For simplicity, just use the raw gradient
            return max_ov, grad
        
        # Multiple restarts with L-BFGS-B
        n_restarts = 3
        for r in range(n_restarts):
            if elapsed() > deadline - 5:
                break
            
            if r == 0:
                x_start = best_h.copy()
            else:
                # Perturb
                x_start = best_h + np.random.randn(N) * 0.05
                x_start = np.clip(x_start, 0, 1)
            
            try:
                res = minimize(
                    lambda x: obj_grad(x)[0],
                    x_start,
                    jac=lambda x: obj_grad(x)[1],
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * N,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < best_score:
                    h_res = np.clip(res.x, 0, 1)
                    # Normalize
                    s = h_res.sum()
                    if s > 0:
                        h_res = h_res * (N/2.0) / s
                    h_res = np.clip(h_res, 0, 1)
                    s2 = h_res.sum()
                    if s2 > 0:
                        h_res = h_res * (N/2.0) / s2
                    score, _ = overlap_max(h_res)
                    if score < best_score:
                        best_score = score
                        best_h = h_res.copy()
            except Exception:
                pass
    
    # Final: ensure exact sum constraint
    h_final = np.clip(best_h, 0, 1)
    s = h_final.sum()
    if s > 0:
        h_final = h_final * (N/2.0) / s
    # After scaling, some values might exceed 1 or go below 0
    h_final = np.clip(h_final, 0, 1)
    # One more normalization pass
    s = h_final.sum()
    if s > 0:
        h_final = h_final * (N/2.0) / s
    h_final = np.clip(h_final, 0, 1)
    
    # Verify sum is close to N/2
    # If not exact, do a final adjustment
    diff = N/2.0 - h_final.sum()
    if abs(diff) > 1e-10:
        # Distribute the difference evenly
        h_final += diff / N
        h_final = np.clip(h_final, 0, 1)
    
    return h_final
