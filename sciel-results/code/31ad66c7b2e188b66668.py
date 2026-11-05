import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Base vector
    base = np.ones(N)
    
    best_h = base.copy()
    best_val = np.inf
    
    # Generate several candidate perturbations
    num_candidates = 50
    
    for _ in range(num_candidates):
        if time.time() > deadline - 5:
            break
            
        # Generate random symmetric sequence
        # First half
        half = np.random.choice([-1, 1], size=N//2)
        # Make symmetric
        s = np.concatenate([half, half[::-1]])
        
        # Make zero mean
        s_mean = np.mean(s)
        s_zero = s - s_mean
        
        # Compute autoconvolution of s_zero
        # We only need to add it to base conv
        # But we need to optimize alpha
        # C_k(h) = C_k(base) + alpha * C_k(s_zero)
        # C_k(base) is triangle T
        
        # Compute C(s_zero)
        c_s = np.convolve(s_zero, s_zero)
        
        # Compute T (conv of base)
        # T_k = N - |k - (N-1)| for k in [0, 2N-2]
        k = np.arange(2*N - 1)
        T = N - np.abs(k - (N - 1))
        
        # We want to minimize max( T + alpha * c_s )
        # Let f(alpha) = max( T + alpha * c_s )
        # This is a convex function of alpha (max of linear functions)
        # We can find the minimum using ternary search or just grid search
        # Range of alpha?
        # If alpha is too large, some terms become large negative or positive.
        # T is positive. c_s can be pos or neg.
        # We expect alpha to be small, around -1 to 1
        
        # Grid search for alpha
        alphas = np.linspace(-2, 2, 401)
        
        min_max_val = np.inf
        best_alpha = 0
        
        for a in alphas:
            curr_conv = T + a * c_s
            max_val = np.max(curr_conv)
            if max_val < min_max_val:
                min_max_val = max_val
                best_alpha = a
                
        # Calculate the actual value for this h
        # value = abs(2 * N * max_conv / sum_h^2)
        # sum_h = sum(1 + a * s_zero) = N + a * 0 = N
        # So value = 2 * N * min_max_val / N^2 = 2 * min_max_val / N
        
        val = 2 * min_max_val / N
        
        if val < best_val:
            best_val = val
            best_h = base + best_alpha * s_zero
            
    return best_h
