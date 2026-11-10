import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    N = 10_000_000
    
    # Sieve for sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    # Search
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    K = 50  # check a_base ± K
    
    # Vectorized search
    b_arr = np.arange(N, 0, -1, dtype=np.int64)
    a_base = (3 * b_arr) // 2
    
    for k in range(-K, K + 1):
        if time.time() > deadline:
            break
        a_arr = a_base + k
        valid = (a_arr >= 1) & (a_arr <= N)
        if not np.any(valid):
            continue
        a_valid = a_arr[valid]
        b_valid = b_arr[valid]
        # Check sigma match
        matches = sigma[a_valid] == sigma[b_valid]
        if not np.any(matches):
            continue
        a_match = a_valid[matches]
        b_match = b_valid[matches]
        # Compute k_val = |2a - 3b|
        k_vals = np.abs(2 * a_match - 3 * b_match)
        # Filter out k_val == 0
        nonzero = k_vals > 0
        if not np.any(nonzero):
            continue
        k_vals = k_vals[nonzero]
        b_match = b_match[nonzero]
        a_match = a_match[nonzero]
        # Compute error
        errs = k_vals.astype(np.float64) / (2.0 * b_match.astype(np.float64))
        min_idx = np.argmin(errs)
        if errs[min_idx] < best_err:
            best_err = errs[min_idx]
            best_a = int(a_match[min_idx])
            best_b = int(b_match[min_idx])
    
    return [int(best_a), int(best_b)]
