import numpy as np
import time

N = 100
TARGET_SUM = N / 2.0

def compute_value(h):
    # h is a numpy array of length N
    # corr = np.correlate(h, 1 - h, mode="full")
    # value = max(corr) / N * 2
    # To speed up, we can compute manually or just use numpy.
    # N=100 is small, numpy correlate is fast.
    c = np.correlate(h, 1.0 - h, mode="full")
    return np.max(c) / N * 2.0

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for 120s limit
    
    best_h = None
    best_val = float('inf')
    
    # Initial best: Seed
    h_seed = np.full(N, 0.5)
    best_val = compute_value(h_seed)
    best_h = h_seed.copy()
    
    # Try to find a better binary sequence
    # Random binary sequences
    num_restarts = 1000
    max_iter_per_restart = 2000
    
    # We will use a simple local search on binary sequences first
    # Then maybe refine with real values?
    # Let's stick to binary swaps first as it's discrete and fast to evaluate changes?
    # Actually, evaluating full correlation every time is O(N^2).
    # 2000 * 1000 = 2,000,000 evals.
    # Each eval ~ 100*100 = 10,000 ops.
    # 2 * 10^10 ops might be too slow in Python if not optimized.
    # Numpy correlate is C-speed. 2M calls to numpy correlate might be slow due to overhead.
    # Let's check speed.
    # 1 call to correlate(100, 100) is ~ 1-5 microseconds?
    # 2M calls * 5us = 10 seconds. This is feasible.
    
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize random binary sequence
        h = np.zeros(N)
        h[:N//2] = 1.0
        
        # Shuffle
        rng.shuffle(h)
        
        current_val = compute_value(h)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        # Local Search: Swap
        # Temperature for SA? Or just Hill Climbing with random restarts?
        # Hill climbing gets stuck. SA is better.
        T = 0.01
        min_T = 0.001
        
        for i in range(max_iter_per_restart):
            if time.time() - start_time > timeout:
                break
                
            # Pick two random indices
            idx1 = rng.integers(N)
            idx2 = rng.integers(N)
            
            if idx1 == idx2:
                continue
                
            # Swap
            h[idx1], h[idx2] = h[idx2], h[idx1]
            
            new_val = compute_value(h)
            
            delta = new_val - current_val
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                # Swap back
                h[idx1], h[idx2] = h[idx2], h[idx1]
                
            T = max(min_T, T * 0.999) # Cool down
            
    # After binary search, try to refine with real values?
    # If best_val is still high, maybe real values help.
    # But binary is a strong candidate.
    
    # One more check: ensure sum is correct
    # Binary sequences have sum exactly 50.
    
    return best_h
