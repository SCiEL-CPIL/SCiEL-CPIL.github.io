import numpy as np
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50  # 50s margin for 60s limit
    
    # Compute Largest Prime Factor for each number up to N
    # Using a sieve approach
    lpf = np.ones(N + 1, dtype=np.int32)
    
    # Sieve for largest prime factors
    # We iterate i from 2 to N. If lpf[i] is 1, i is prime.
    # Then we set lpf[j] = i for all multiples j of i.
    # Since we go in increasing order, the last assignment for any j
    # will be from its largest prime factor.
    for i in range(2, N + 1):
        if lpf[i] == 1:
            # i is prime
            lpf[i::i] = i
            
    # Check time
    if time.time() > deadline:
        # Fallback: return a valid pair, e.g., 1 and N-1
        return [1, N - 1]

    # Vectorized scan to find min max(P(a), P(b))
    a = np.arange(1, N, dtype=np.int32)
    b = N - a
    
    pa = lpf[a]
    pb = lpf[b]
    
    # Calculate max prime factor for each pair
    m = np.maximum(pa, pb)
    
    # Find the index with the minimum value
    idx = np.argmin(m)
    
    best_a = int(a[idx])
    best_b = int(b[idx])
    
    return [best_a, best_b]
