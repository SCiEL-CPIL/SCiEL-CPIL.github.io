import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer for 120s limit
    
    # Strategy: Sieve-based sigma computation, find M with most k solutions
    # K_max chosen to complete within time budget
    K_max = 5_000_000
    
    # Compute sigma(k) for k = 1..K_max using the divisor sieve
    sigma = np.zeros(K_max + 1, dtype=np.int64)
    
    # Perform the sieve with time guard
    for d in range(1, K_max + 1):
        if time.time() > deadline - 30.0:
            K_max = d - 1
            break
        sigma[d::d] += d
    
    # Calculate M = k * sigma(k)
    k_arr = np.arange(1, K_max + 1, dtype=np.int64)
    M_arr = k_arr * sigma[1:]
    
    # Filter M <= 10^18
    valid_mask = M_arr <= 10**18
    
    M_valid = M_arr[valid_mask]
    k_valid = k_arr[valid_mask]
    
    if len(M_valid) == 0:
        return {'M': 1, 'k': [1]}
    
    # Find the M with the most k values
    unique_M, inverse = np.unique(M_valid, return_inverse=True)
    counts = np.bincount(inverse)
    
    # Find the top M value by count
    max_idx = np.argmax(counts)
    max_M = unique_M[max_idx]
    
    k_solutions = k_valid[inverse == max_idx]
    k_list = [int(k) for k in k_solutions.tolist()]
    
    if not k_list:
        return {'M': 1, 'k': [1]}
        
    # Verify all solutions satisfy k_i <= isqrt(M)
    sqrt_M = math.isqrt(int(max_M))
    k_list = [k for k in k_list if k <= sqrt_M]
    
    if not k_list:
        return {'M': 1, 'k': [1]}
        
    return {'M': int(max_M), 'k': k_list}
