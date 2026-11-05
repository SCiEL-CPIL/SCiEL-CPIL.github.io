"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def construct():
    # We want to minimize V = | 2 * N * max(conv(h, h)) / (sum(h))^2 |
    # A good candidate is a step function: 1 for i < K, -1 for i >= K.
    # Let's search for the optimal K.
    
    best_h = None
    best_val = float('inf')
    
    # The optimal K for the step function is around N/4.
    # We scan a reasonable range around N/4 to be safe and fast.
    # N/4 = 100.
    # We'll check K from N//4 - 50 to N//4 + 50.
    
    start_k = max(1, N // 4 - 50)
    end_k = min(N - 1, N // 4 + 50)
    
    for K in range(start_k, end_k + 1):
        h = np.empty(N)
        h[:K] = 1.0
        h[K:] = -1.0
        
        s = np.sum(h)
        if s == 0:
            continue
            
        # Compute autoconvolution
        # For N=400, np.convolve is fast enough.
        # max is taken over signed entries.
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        
        val = abs(2 * N * max_conv / (s * s))
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Fallback to seed if something goes wrong
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
