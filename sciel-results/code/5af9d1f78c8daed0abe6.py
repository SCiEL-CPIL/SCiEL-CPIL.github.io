import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 110.0  # 120s limit, keep margin
    
    # Initial guess: alternating 1s and 0s, then shuffle slightly or use random
    # Actually, a good initial guess for minimizing max correlation is to avoid regular patterns.
    # Let's start with a random binary sequence with sum 50.
    
    best_h = None
    best_val = np.inf
    
    # Try a few random restarts
    for _ in range(100):
        if time.time() > deadline:
            break
            
        # Random binary sequence with 50 ones
        h = np.zeros(N)
        indices = np.random.choice(N, N//2, replace=False)
        h[indices] = 1.0
        
        # Local search: swap pairs to reduce max correlation
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Calculate current max correlation
            # To speed up, we can just compute the full correlation
            corr = np.correlate(h, 1 - h, mode="full")
            max_corr = np.max(corr)
            
            # Try random swaps
            for _ in range(1000):
                if time.time() > deadline:
                    break
                i, j = np.random.choice(N, 2, replace=False)
                if h[i] == h[j]:
                    continue
                
                # Swap
                h[i], h[j] = h[j], h[i]
                
                # Calculate new max correlation
                # We can update correlation incrementally or just recompute
                # Recomputing is O(N^2) ~ 10^4, which is fast.
                new_corr = np.correlate(h, 1 - h, mode="full")
                new_max = np.max(new_corr)
                
                if new_max < max_corr:
                    max_corr = new_max
                    improved = True
                else:
                    # Swap back
                    h[i], h[j] = h[j], h[i]
                    
        val = max_corr / N * 2
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    if best_h is None:
        return np.full(N, 0.5)
        
    return best_h
