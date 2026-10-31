"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """
    Calculate the maximum linear overlap of h with shifted 1-h.
    h is an array of 0s and 1s.
    """
    # h and 1-h are real valued
    # np.correlate(a, v, mode='full')
    # c_k = sum_n a[n+k] * v[n]
    # We want sum_i h[i] * (1-h[i-k])
    # Let a = h, v = 1-h.
    # c_k = sum_i h[i] * (1-h[i-k])
    # This corresponds to lag k in correlate(h, 1-h)
    # The full output has length 2*N - 1.
    # Index N-1 corresponds to lag 0.
    # Index N-1 + k corresponds to lag k.
    corr = np.correlate(h, 1.0 - h, mode='full')
    # We only care about non-negative lags k=0..N-1
    # These are indices N-1 to 2*N-2
    overlaps = corr[N-1 : 2*N-1]
    return np.max(overlaps)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Baseline: constant 0.5
    h_const = np.full(N, 0.5)
    best_score = _max_overlap(h_const)
    best_h = h_const
    
    # Random search over 0/1 sequences with exactly 50 ones
    base = np.array([1.0]*50 + [0.0]*50)
    
    iterations = 0
    while time.time() < deadline:
        h = base.copy()
        np.random.shuffle(h)
        score = _max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h
        iterations += 1
        
        # If we find a very good score, we might want to polish?
        # For now, just keep searching.
        
    return best_h
# EVOLVE-BLOCK-END
