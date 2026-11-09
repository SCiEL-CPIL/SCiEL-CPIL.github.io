import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p : limit+1 : p] = False
    
    primes = np.where(is_prime)[0]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [int(primes[0])]
    
    # dp[j][i] = length of longest valid sequence ending with primes[j], primes[i]
    # j < i
    dp = np.zeros((N, N), dtype=np.int16)
    parent = np.full((N, N), -1, dtype=np.int32)
    
    global_max_len = 2
    best_j = 0
    best_i = 1
    if N > 1:
        dp[0][1] = 2
        # parent[0][1] is -1 by default
    
    for j in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
            
        pj = primes[j]
        
        # Prepare suffix max and argmax for dp[k][j] where k < j
        # prev_vals[k] = dp[k][j]
        prev_vals = dp[:j, j]
        
        # Compute suffix max and argmax
        # We want suf_max[k] = max(prev_vals[k], ..., prev_vals[j-1])
        # and arg_suf[k] = index of that max
        suf_max = np.zeros(j, dtype=np.int16)
        arg_suf = np.zeros(j, dtype=np.int32)
        
        current_max = 0
        current_arg = -1
        
        # Iterate from right to left
        for k in range(j - 1, -1, -1):
            if prev_vals[k] > current_max:
                current_max = prev_vals[k]
                current_arg = k
            elif prev_vals[k] == current_max:
                # Prefer larger index? Doesn't matter for validity.
                # But let's stick to the first one found from right (largest index)
                # Actually, if equal, any is fine.
                pass
            
            suf_max[k] = current_max
            arg_suf[k] = current_arg
            
        # Now iterate i > j
        # For each i, find K = searchsorted(primes[:j], 2*pj - pi)
        # If K < j, dp[j][i] = suf_max[K] + 1, parent[j][i] = arg_suf[K]
        # Else dp[j][i] = 2
        
        # Vectorize the search?
        # thresholds = 2 * pj - primes[j+1:]
        # Ks = np.searchsorted(primes[:j], thresholds, side='left')
        # valid = Ks < j
        # ...
        
        # Let's try vectorized approach for speed
        if j + 1 < N:
            targets = primes[j+1:]
            thresholds = 2 * pj - targets
            Ks = np.searchsorted(primes[:j], thresholds, side='left')
            
            valid_mask = Ks < j
            
            # For valid ones
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                # Corresponding i indices in primes
                i_indices = j + 1 + valid_indices
                
                # Get values
                vals = suf_max[Ks[valid_indices]] + 1
                
                dp[j, i_indices] = vals
                parent[j, i_indices] = arg_suf[Ks[valid_indices]]
                
                # Update global max
                max_val = np.max(vals)
                if max_val > global_max_len:
                    global_max_len = max_val
                    # Find the index
                    idx_in_valid = np.argmax(vals)
                    best_j = j
                    best_i = int(i_indices[idx_in_valid])
            
            # For invalid ones, dp is 2
            if np.any(~valid_mask):
                invalid_indices = np.where(~valid_mask)[0]
                i_indices_inv = j + 1 + invalid_indices
                dp[j, i_indices_inv] = 2
                # parent remains -1
    
    # Reconstruct
    if global_max_len == 1:
        return [int(primes[0])]
        
    seq = []
    curr_j = best_j
    curr_i = best_i
    
    seq.append(int(primes[curr_i]))
    seq.append(int(primes[curr_j]))
    
    k = parent[curr_j][curr_i]
    while k != -1:
        seq.append(int(primes[k]))
        # Move state to (k, curr_j)
        curr_i = curr_j
        curr_j = k
        k = parent[curr_j][curr_i]
        
    seq.reverse()
    return seq
