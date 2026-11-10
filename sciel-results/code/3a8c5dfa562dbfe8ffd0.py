import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Sieve limit
    L = 20_000_000
    
    # Compute sigma(k) for k <= L
    sigma = np.ones(L + 1, dtype=np.int64)
    
    # Split loop to reduce Python overhead for i > L/2
    half_L = L // 2
    for i in range(1, half_L + 1):
        sigma[i::i] += i
        
    # For i > L/2, sigma[i] += i is the only update
    # sigma[half_L+1:] += np.arange(half_L+1, L+1, dtype=np.int64)
    # This vectorized addition is fast.
    sigma[half_L + 1:] += np.arange(half_L + 1, L + 1, dtype=np.int64)
    
    # Compute M = k * sigma(k)
    k_vals = np.arange(1, L + 1, dtype=np.int64)
    M_vals = k_vals * sigma
    
    # Filter M <= 10^18
    max_M = 10**18
    mask = M_vals <= max_M
    M_valid = M_vals[mask]
    k_valid = k_vals[mask]
    
    # Group by M
    # Using a dictionary
    groups = {}
    for m, k in zip(M_valid, k_valid):
        if m in groups:
            groups[m].append(k)
        else:
            groups[m] = [k]
            
    # Find the best M with the most valid k's
    best_M = 0
    best_k = []
    
    for m, ks in groups.items():
        if len(ks) < 2:
            continue
        sqrt_m = math.isqrt(m)
        # Filter k <= sqrt(m)
        # Since k is sorted in the list? No, k_valid is sorted, but groups append in order.
        # So ks is sorted.
        # We can just filter.
        valid_ks = [k for k in ks if k <= sqrt_m]
        
        if len(valid_ks) > len(best_k):
            best_k = valid_ks
            best_M = m
            
    # Fallback if no solution found with >= 2
    if not best_k:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
