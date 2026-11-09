import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    n = len(primes)
    primes_arr = np.array(primes)
    
    prime_to_idx = np.full(limit + 1, -1, dtype=int)
    for i, p in enumerate(primes):
        prime_to_idx[p] = i
        
    G = primes[-1] - primes[0]
    
    # f[j][g] = length of longest subsequence ending at primes[j] with last gap exactly g
    f = np.zeros((n, G + 1), dtype=int)
    # M[j][g] = max length of subsequence ending at primes[j] with last gap <= g
    M = np.zeros((n, G + 1), dtype=int)
    # argmax_prefix[j][g] = gap g' <= g such that f[j][g'] = M[j][g]
    argmax_prefix = np.zeros((n, G + 1), dtype=int)
    # prev_idx[j][g] = index i such that f[j][g] comes from M[i][g]
    prev_idx = np.full((n, G + 1), -1, dtype=int)
    # prev_gap[j][g] = gap g' such that the previous state is (i, g')
    prev_gap = np.full((n, G + 1), -1, dtype=int)
    
    max_len = 0
    best_j = -1
    best_g = -1
    
    g_vals = np.arange(1, G + 1)
    
    for j in range(n):
        if time.time() > deadline:
            break
            
        p_j = primes[j]
        p_prevs = p_j - g_vals
        valid_mask = (p_prevs >= 2) & (p_prevs <= limit)
        
        if np.any(valid_mask):
            indices = prime_to_idx[p_prevs[valid_mask]]
            valid_mask_indices = indices >= 0
            
            if np.any(valid_mask_indices):
                idx_valid = indices[valid_mask_indices]
                g_valid = g_vals[valid_mask][valid_mask_indices]
                
                M_valid = M[idx_valid, g_valid]
                f_valid = 1 + np.maximum(1, M_valid)
                
                # Map back to valid_mask
                # We need to set f[j, valid_mask] at the correct positions
                # valid_mask is a boolean array of length G
                # valid_mask_indices is a boolean array of length sum(valid_mask)
                # This is tricky to index directly.
                
                # Let's create a full array of indices
                all_indices = prime_to_idx[p_prevs]
                all_indices[~valid_mask] = -1
                valid_all = (all_indices >= 0)
                
                if np.any(valid_all):
                    idx_v = all_indices[valid_all]
                    g_v = g_vals[valid_all]
                    
                    M_v = M[idx_v, g_v]
                    f_v = 1 + np.maximum(1, M_v)
                    
                    f[j, valid_all] = f_v
                    prev_idx[j, valid_all] = idx_v
                    prev_gap[j, valid_all] = argmax_prefix[idx_v, g_v]
                    
                    current_max = np.max(f[j])
                    if current_max > max_len:
                        max_len = current_max
                        best_j = j
                        # Find the gap that gave the max
                        # There might be multiple, pick one
                        best_g = np.argmax(f[j])
        
        M[j] = np.maximum.accumulate(f[j])
        
        # Compute argmax_prefix[j]
        f_j = f[j]
        M_j = M[j]
        # Find indices where f_j == M_j and f_j > 0
        match_mask = (f_j == M_j) & (f_j > 0)
        match_idx = np.where(match_mask)[0]
        
        if len(match_idx) > 0:
            pos = np.searchsorted(match_idx, np.arange(G + 1), side='right')
            arg_j = np.where(pos > 0, match_idx[pos - 1], 0)
        else:
            arg_j = np.zeros(G + 1, dtype=int)
            
        argmax_prefix[j] = arg_j

    if max_len == 0:
        return [primes[0]] if n > 0 else []
        
    # Backtrack
    seq = []
    curr_j = best_j
    curr_g = best_g
    seq.append(primes[curr_j])
    
    while prev_idx[curr_j, curr_g] != -1:
        prev_j = prev_idx[curr_j, curr_g]
        prev_g = prev_gap[curr_j, curr_g]
        seq.append(primes[prev_j])
        curr_j = prev_j
        curr_g = prev_g
        
    seq.reverse()
    return seq

