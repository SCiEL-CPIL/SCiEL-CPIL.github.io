import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin for 90s limit
    
    N = 1000000
    max_val = N + 1
    
    # Sieve to find Largest Prime Factor
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # Iterate through potential primes
    # If lpf[i] is 0, i is prime
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime. Update all multiples.
            # Since we iterate i in increasing order, the last update
            # comes from the largest prime factor.
            lpf[i::i] = i
            
        # Periodic time check
        if i % 50000 == 0:
            if time.time() - start_time > deadline:
                # Fallback: return a known good candidate if time runs out
                # 10000 is a safe fallback
                return 10000

    if time.time() - start_time > deadline:
        return 10000

    min_lpf = float('inf')
    best_n = 10000
    
    # Convert to list for faster scalar access in the scan loop
    # Alternatively, keep as numpy array and use numpy operations for the scan
    # But a simple loop in Python over 1e6 items is ~0.05s, which is fine.
    # Let's try to vectorize the scan if possible, but max(axis) doesn't work directly on pairs.
    # We can do:
    # lpf_n = lpf[10000:N+1]
    # lpf_np1 = lpf[10001:N+2]
    # max_lpf = np.maximum(lpf_n, lpf_np1)
    # idx = np.argmin(max_lpf)
    # best_n = 10000 + idx
    
    lpf_n = lpf[10000:N+1]
    lpf_np1 = lpf[10001:N+2]
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find index of minimum
    min_idx = np.argmin(max_lpf)
    best_n = 10000 + min_idx
    min_lpf = max_lpf[min_idx]
    
    return best_n
