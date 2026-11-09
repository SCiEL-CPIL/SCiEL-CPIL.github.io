import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 80.0
    
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
    
    # Initialize base cases: sequences of length 2
    # dp[j][i] = 2 for all j < i
    # We can do this in the loop or pre-fill.
    # Pre-filling is faster.
    for j in range(N):
        if j + 1 < N:
            dp[j, j+1:] = 2
    
    global_max_len = 2
    best_j = 0
    best_i = 1
    
    for j in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
            
        pj = primes[j]
        
        # prev_vals[k] = dp[k][j] for k in 0..j-1
        prev_vals = dp[:j, j]
        
        # Compute suffix max and argmax for prev_vals
        # suf_max[k] = max(prev_vals[k], ..., prev_vals[j-1])
        # arg_suf[k] = index of that max (first occurrence)
        rev_prev = prev_vals[::-1]
        cummax = np.maximum.accumulate(rev_prev)
        suf_max = cummax[::-1]
        
        # Compute arg_suf by scanning right to left
        arg_suf = np.empty(j, dtype=np.int32)
        current_max = -1
        current_arg = -1
        for k in range(j - 1, -1, -1):
            val = prev_vals[k]
            if val > current_max:
                current_max = val
                current_arg = k
            arg_suf[k] = current_arg
            
        # Now compute dp[j][i] for all i > j
        if j + 1 < N:
            targets = primes[j+1:]
            thresholds = 2 * pj - targets
            Ks = np.searchsorted(primes, thresholds, side='left')
            
            valid_mask = Ks < j
            
            if np.any(valid_mask):
                # Extract valid parts
                k_indices = Ks[valid_mask]
                vals = suf_max[k_indices] + 1
                args = arg_suf[k_indices]
                
                # Find indices in the full array
                i_indices = j + 1 + np.where(valid_mask)[0]
                
                dp[j, i_indices] = vals
                parent[j, i_indices] = args
                
                # Update global max
                max_val = np.max(vals)
                if max_val > global_max_len:
                    global_max_len = max_val
                    idx_in_valid = np.argmax(vals)
                    best_j = j
                    best_i = int(i_indices[idx_in_valid])
    
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
