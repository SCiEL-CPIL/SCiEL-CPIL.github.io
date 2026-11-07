import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    # Use divisor-sum sieve with numpy for efficiency
    # sigma_arr[d::d] += d is a single vectorized operation per d
    N = 15_000_000
    
    sigma_arr = np.zeros(N + 1, dtype=np.int64)
    
    # Divisor sum sieve: for each d, add d to all multiples of d
    for d in range(1, N + 1):
        if d % 200000 == 0:
            if time.time() > deadline - 30:
                break
        sigma_arr[d::d] += d
    
    # Compute M = k * sigma(k) and find best collision
    groups = {}
    limit = 10**18
    
    # Use numpy to compute all M values at once for speed
    k_arr = np.arange(1, N + 1, dtype=np.int64)
    M_arr = k_arr * sigma_arr[1:N+1]
    
    # Filter M <= limit
    valid_mask = M_arr <= limit
    valid_M = M_arr[valid_mask]
    valid_k = k_arr[valid_mask]
    
    # Group by M value using numpy
    # Use dictionary approach for grouping
    for i in range(len(valid_M)):
        if i % 2000000 == 0 and time.time() > deadline - 15:
            break
        m_val = int(valid_M[i])
        k_val = int(valid_k[i])
        if m_val in groups:
            groups[m_val].append(k_val)
        else:
            groups[m_val] = [k_val]
    
    # Find best M
    best_M = 0
    best_k_list = []
    max_count = 0
    
    for M, k_list in groups.items():
        if len(k_list) > max_count:
            max_count = len(k_list)
            best_M = M
            best_k_list = k_list
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k_list = [int(k) for k in best_k_list]
    
    return {'M': best_M, 'k': best_k_list}
