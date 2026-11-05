import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    MAX_N = 1000000
    MAX_V = MAX_N + 1

    # Initialize LPF array
    lpf = np.zeros(MAX_V + 1, dtype=np.int64)
    
    # Sieve for Largest Prime Factor using NumPy
    # We iterate through potential primes.
    # If lpf[i] == 0, i is prime.
    # We update all multiples of i.
    
    # Optimization: We can skip even numbers > 2 after handling 2, but simple loop is fine with NumPy slicing.
    # However, Python loop over 1e6 is ~0.1s if body is fast.
    # The inner loop is vectorized assignment.
    
    for i in range(2, MAX_V + 1):
        if time.time() > deadline:
            break
        
        if lpf[i] == 0:
            # i is prime
            # Update multiples
            # lpf[i::i] = i
            # This is a vectorized operation.
            lpf[i::i] = i
            
    # Search for n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # Vectorized search
    # We need to check n from 10000 to 1000000.
    # lpf_n = lpf[10000:MAX_N+1]
    # lpf_np1 = lpf[10001:MAX_N+2]
    
    n_start = 10000
    n_end = MAX_N
    
    # Slice the LPF array
    # Note: lpf is 0-indexed.
    # We want indices n_start to n_end inclusive for n.
    # So lpf[n_start : n_end + 1]
    # And lpf[n_start + 1 : n_end + 2]
    
    lp_n = lpf[n_start:n_end + 1]
    lp_np1 = lpf[n_start + 1:n_end + 2]
    
    # Compute max for each pair
    max_lpf = np.maximum(lp_n, lp_np1)
    
    # Find the index of the minimum
    min_idx = np.argmin(max_lpf)
    min_max_lpf = max_lpf[min_idx]
    best_n = n_start + min_idx
    
    # Fallback if something went wrong
    if best_n < n_start or best_n > n_end:
        return 10000
        
    return int(best_n)
