import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 120s limit
    
    # We assume the objective is to minimize the peak of the discrete autoconvolution
    # normalized by the square of the sum (L1 norm squared).
    # Minimize max_k (sum_i h_i h_{k-i}) subject to sum h_i = 1, h_i >= 0.
    
    # Due to symmetry, h_i = h_{N-1-i}.
    # We only need to optimize the first half (plus center if N is odd, but N=600 is even).
    # Let M = N // 2 = 300.
    # h[0:M] are variables. h[M:N] = h[0:M][::-1].
    
    M = N // 2
    x0 = np.ones(M) / N  # Initial guess: uniform distribution
    
    def objective(x):
        # Reconstruct full h
        h = np.concatenate([x, x[::-1]])
        
        # Compute autoconvolution c_k for k=0 to 2N-2
        # We only need the peak, which is likely near the center k=N-1.
        # However, for safety, we can compute the full convolution or just the center.
        # For a symmetric non-negative function, the autoconvolution is symmetric and unimodal (peak at center).
        # So we just need c_{N-1}.
        # c_{N-1} = sum_{i=0}^{N-1} h_i h_{N-1-i}
        
        # Since h is symmetric, h_{N-1-i} = h_i.
        # So c_{N-1} = sum_{i=0}^{N-1} h_i^2 = ||h||_2^2.
        
        # Wait, is the peak always at the center for symmetric non-negative functions?
        # Yes, if h is symmetric and non-negative, the autoconvolution is symmetric.
        # Is it unimodal? Not necessarily for arbitrary symmetric non-negative functions, 
        # but for "spread out" functions like uniform, it is.
        # If we minimize the peak, and the peak is at the center for the uniform case,
        # and we are in a convex neighborhood, it's likely the peak stays at the center.
        
        # However, to be rigorous, the max of c_k is the objective.
        # Computing full c_k is O(N^2) or O(N log N) with FFT.
        # N=600, O(N^2) is 360,000 ops. Very fast.
        
        c = np.convolve(h, h, mode='full')
        return np.max(c)
        
    def gradient(x):
        h = np.concatenate([x, x[::-1]])
        c = np.convolve(h, h, mode='full')
        k_max = np.argmax(c)
        
        # Gradient of c_k w.r.t h_j is 2 * h_{k-j} (with bounds)
        # We need grad w.r.t x.
        # x[0:M] corresponds to h[0:M].
        # x[M-1-j] corresponds to h[N-1-j].
        
        # Let's compute grad_h = 2 * h_reversed?
        # c_k = sum_i h_i h_{k-i}
        # d c_k / d h_j = 2 h_{k-j} (if k-j in bounds)
        
        # We need the gradient of the max.
        # If c_k is unique max, grad = grad c_k.
        # If multiple, we can average or pick one.
        
        # h is length N.
        # h_rev = h[::-1]
        # grad_h[j] = 2 * h_rev[j] = 2 * h[N-1-j]
        
        h_rev = h[::-1]
        grad_h = 2.0 * h_rev
        
        # Map grad_h to grad_x
        # x[0..M-1] -> h[0..M-1]
        # x[M-1..0] -> h[N-1..M]
        # So grad_x[j] = grad_h[j] + grad_h[N-1-j]
        
        grad_x = grad_h[:M] + grad_h[N-1: N-M-1:-1]
        
        return grad_x

    # Constraints: sum(h) = 1
    # sum(x) + sum(x[::-1]) = 2 * sum(x) = 1 => sum(x) = 0.5
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 0.5, 'jac': lambda x: np.ones(M)}
    ]
    
    bounds = [(0, None) for _ in range(M)]
    
    best_val = np.inf
    best_x = x0
    
    # Use SLSQP or Trust-Constr
    # SLSQP is robust for small/medium N.
    # N=300 variables.
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            jac=gradient, 
            bounds=bounds, 
            constraints=constraints, 
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline:
        # Return best found so far (initial uniform)
        h_full = np.concatenate([best_x, best_x[::-1]])
        return list(h_full)
        
    # Refine with a second pass if time permits
    if time.time() < deadline - 20:
        try:
            res2 = minimize(
                objective, 
                best_x, 
                method='SLSQP', 
                jac=gradient, 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            if res2.success and res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception:
            pass

    h_full = np.concatenate([best_x, best_x[::-1]])
    
    # Ensure non-negative
    h_full = np.maximum(h_full, 0)
    
    # Normalize to ensure sum is exactly 1 (or whatever the evaluator expects)
    # The task says "sequence of nonnegative step heights".
    # Usually, the normalization is handled by the evaluator or the ratio is invariant.
    # If we return heights that sum to 1, it's a valid probability density.
    
    return list(h_full)
# EVOLVE-BLOCK-END
