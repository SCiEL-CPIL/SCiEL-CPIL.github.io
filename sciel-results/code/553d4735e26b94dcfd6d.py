import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s deadline to be safe within 60s
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in 1..N
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes variant for sigma
    # for d in 1..N: for m in d, 2d, ...: sigma[m] += d
    # Vectorized:
    for d in range(1, N + 1):
        if time.time() > deadline - 5:
            break
        sigma[d::d] += d
        
    # 2. Find best pair
    b = np.arange(1, N + 1, dtype=np.int64)
    
    # Candidates for a near 1.5 * b
    # 1.5 * b = 3 * b / 2
    # If 3*b is even, a = 3*b//2
    # If 3*b is odd, a = 3*b//2 and a = 3*b//2 + 1
    
    a1 = (3 * b) // 2
    a2 = a1 + 1
    
    # We must ensure a <= N
    valid_a1 = a1 <= N
    valid_a2 = a2 <= N
    
    # Check sigma equality
    # Note: sigma[0] is 0, but b >= 1, a >= 1, so no index 0 issue.
    
    # For a1
    # We need sigma[a1] == sigma[b]
    # Only check where valid_a1 is True to avoid index errors? 
    # Actually a1 is always >= 1 for b>=1.
    # But if a1 > N, we can't index sigma.
    # So we create masks.
    
    match1 = np.zeros(N, dtype=bool)
    match2 = np.zeros(N, dtype=bool)
    
    # Efficient indexing
    # We can't just do sigma[a1] if a1 has values > N.
    # So we filter first.
    
    idx1 = np.where(valid_a1)[0]
    if len(idx1) > 0:
        a1_vals = a1[idx1]
        b_vals = b[idx1]
        match1[idx1] = (sigma[a1_vals] == sigma[b_vals])
        
    idx2 = np.where(valid_a2)[0]
    if len(idx2) > 0:
        a2_vals = a2[idx2]
        b_vals = b[idx2]
        match2[idx2] = (sigma[a2_vals] == sigma[b_vals])
        
    # Calculate errors
    # Error = |a/b - 1.5|
    # We want to minimize this.
    
    best_error = float('inf')
    best_a = 0
    best_b = 0
    
    # Check matches for a1
    if np.any(match1):
        idxs = np.where(match1)[0]
        a_vals = a1[idxs]
        b_vals = b[idxs]
        # Calculate error
        # Use float division
        errors = np.abs(a_vals.astype(np.float64) / b_vals.astype(np.float64) - 1.5)
        min_idx = np.argmin(errors)
        min_err = errors[min_idx]
        if min_err < best_error:
            best_error = min_err
            best_a = int(a_vals[min_idx])
            best_b = int(b_vals[min_idx])
            
    # Check matches for a2
    if np.any(match2):
        idxs = np.where(match2)[0]
        a_vals = a2[idxs]
        b_vals = b[idxs]
        errors = np.abs(a_vals.astype(np.float64) / b_vals.astype(np.float64) - 1.5)
        min_idx = np.argmin(errors)
        min_err = errors[min_idx]
        if min_err < best_error:
            best_error = min_err
            best_a = int(a_vals[min_idx])
            best_b = int(b_vals[min_idx])
            
    # Fallback if no pair found (should not happen)
    if best_a == 0:
        return [11, 6]
        
    return [best_a, best_b]
