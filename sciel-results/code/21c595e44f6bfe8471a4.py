import bisect
import time
from math import isqrt
import numpy as np

def construct():
    deadline = time.time() + 75  # Safety margin before 90s kill
    
    # Generate primes up to 10000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    P = get_primes(10000)
    P_arr = np.array(P, dtype=np.int64)
    N = len(P)
    
    # dp[j][i] = length of longest valid subsequence ending with P[j], P[i] (j < i)
    dp = np.zeros((N, N), dtype=np.int32)
    par = np.full((N, N), -1, dtype=np.int32)
    
    max_len = 0
    best_j = -1
    best_i = -1
    
    # Iterate j from 0 to N-1
    for j in range(N):
        if time.time() > deadline:
            break
        if j == 0:
            continue
            
        # Prepare suffix max for column j (rows 0 to j-1)
        # suf_val[k] = max(dp[0..k][j]), suf_idx[k] = argmax index
        suf_val = np.zeros(j, dtype=np.int32)
        suf_idx = np.full(j, -1, dtype=np.int32)
        
        # Initialize last element
        suf_val[j-1] = dp[j-1, j]
        suf_idx[j-1] = j-1
        
        for k in range(j-2, -1, -1):
            if dp[k, j] > suf_val[k+1]:
                suf_val[k] = dp[k, j]
                suf_idx[k] = k
            else:
                suf_val[k] = suf_val[k+1]
                suf_idx[k] = suf_idx[k+1]
                
        # Vectorized inner loop for i from j+1 to N-1
        i_indices = np.arange(j+1, N)
        X = 2 * P_arr[j] - P_arr[i_indices]
        
        # For each i, find k_min = bisect_left(P, X, 0, j)
        # Use searchsorted on P_arr[0:j]
        k_min_arr = np.searchsorted(P_arr[:j], X, side='left')
        
        # Where k_min < j, we have a valid predecessor
        valid_mask = k_min_arr < j
        
        # For valid entries, get the value and predecessor index
        valid_i_idx = np.where(valid_mask)[0]
        if len(valid_i_idx) > 0:
            k_mins = k_min_arr[valid_i_idx]
            vals = suf_val[k_mins]
            k_bests = suf_idx[k_mins]
            
            # Update dp and par for these entries
            actual_i = i_indices[valid_i_idx]
            dp[j, actual_i] = vals + 1
            par[j, actual_i] = k_bests
            
            # Track max
            max_vals = vals + 1
            local_max_idx = np.argmax(max_vals)
            local_max_val = max_vals[local_max_idx]
            if local_max_val > max_len:
                max_len = int(local_max_val)
                best_j = j
                best_i = int(actual_i[local_max_idx])
        
        # For invalid entries, dp remains 0 (meaning no valid chain of length > 1 ending here with this j)
        # But we should set dp[j, i] = 2 for pairs that can't be extended (just the pair itself)
        # Actually, the base case is any pair (P[j], P[i]) has length 2
        # Let's set those that weren't updated to 2
        invalid_i_idx = np.where(~valid_mask)[0]
        if len(invalid_i_idx) > 0:
            actual_i_invalid = i_indices[invalid_i_idx]
            dp[j, actual_i_invalid] = 2
            # par stays -1 (no predecessor beyond the pair)
            
            # Check if 2 > max_len (only matters if max_len < 2)
            if max_len < 2 and len(actual_i_invalid) > 0:
                max_len = 2
                best_j = j
                best_i = int(actual_i_invalid[0])
        elif max_len < 2 and len(valid_i_idx) > 0:
            # All valid, but check if any have value 2
            if np.any(max_vals == 2):
                idx2 = np.where(max_vals == 2)[0][0]
                max_len = 2
                best_j = j
                best_i = int(actual_i[idx2])
    
    # Reconstruct sequence
    if max_len == 0:
        return []
        
    if max_len == 2:
        return [int(P[best_j]), int(P[best_i])]
        
    seq = [int(P[best_i]), int(P[best_j])]
    j_cur = best_j
    i_cur = best_i
    
    while par[j_cur, i_cur] != -1:
        k = int(par[j_cur, i_cur])
        seq.append(int(P[k]))
        i_cur = j_cur
        j_cur = k
        
    seq.reverse()
    return seq
