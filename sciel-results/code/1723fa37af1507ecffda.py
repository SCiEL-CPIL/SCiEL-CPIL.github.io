import time
import numpy as np
from math import isqrt

def construct():
    """
    Finds the longest increasing subsequence of primes <= 10000 
    such that the gaps between consecutive primes are non-decreasing.
    Uses NumPy for accelerated DP transitions.
    """
    deadline = time.time() + 85.0  # Safety margin
    
    # 1. Sieve of Eratosthenes
    n_max = 10000
    is_prime = np.ones(n_max + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(n_max) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.array([i for i in range(2, n_max + 1) if is_prime[i]], dtype=np.int32)
    m = len(primes)
    
    # dp[i][j] = length of longest valid chain ending with primes[i], primes[j] (i < j)
    # parent[i][j] = the index k < i such that dp[i][j] = dp[k][i] + 1
    # If no such k, parent[i][j] = -1
    dp = np.zeros((m, m), dtype=np.int32)
    parent = np.full((m, m), -1, dtype=np.int32)
    
    best_len = 2
    best_i = 0
    best_j = 1
    
    # Iterate over the second-to-last prime index j
    for j in range(m - 1):
        if time.time() > deadline:
            break
            
        # If j == 0, there are no predecessors k < 0
        if j == 0:
            # dp[0][i] = 2 for all i > 0
            dp[0, 1:] = 2
            # parent[0][i] remains -1
            if 2 > best_len:
                best_len = 2
                best_i = 0
                best_j = 1
            continue
        
        # Get dp values for all k < j: dp[k][j]
        prev_vals = dp[:j, j]
        
        # Compute suffix maximum and suffix argmax
        # suf_val[x] = max(prev_vals[x:j])
        # suf_arg[x] = argmax index in [x, j)
        suf_val = np.zeros(j, dtype=np.int32)
        suf_arg = np.zeros(j, dtype=np.int32)
        
        suf_val[j-1] = prev_vals[j-1]
        suf_arg[j-1] = j-1
        
        # Python loop for suffix max/argmax is O(j) and j is small
        for k in range(j-2, -1, -1):
            if prev_vals[k] >= suf_val[k+1]:
                suf_val[k] = prev_vals[k]
                suf_arg[k] = k
            else:
                suf_val[k] = suf_val[k+1]
                suf_arg[k] = suf_arg[k+1]
        
        # Targets for searchsorted: 2*primes[j] - primes[i] for i > j
        targets = 2 * primes[j] - primes[j+1:m]
        
        # Find the first index idx in primes[:j] such that primes[idx] >= target
        # primes[:j] is sorted
        idxs = np.searchsorted(primes[:j], targets, side='left')
        
        # Check if idx is valid (within 0..j-1)
        valid = idxs < j
        
        # Get the max value from suf_val at the found indices
        # If invalid, max value is 0
        # np.minimum(idxs, j-1) ensures we don't index out of bounds if we were to use them, 
        # but we use 'valid' mask.
        # Actually, if idx >= j, it means no prime in [0, j) is >= target.
        # In that case, no predecessor exists.
        
        # We need to map idxs to suf_val.
        # If valid, use suf_val[idxs].
        # If not valid, use 0.
        
        # Create a safe index array
        safe_idxs = np.where(valid, idxs, 0)
        
        max_prevs = np.where(valid, suf_val[safe_idxs], 0)
        preds = np.where(valid, suf_arg[safe_idxs], -1)
        
        # New lengths: max(2, max_prevs + 1)
        new_lengths = np.maximum(2, max_prevs + 1)
        
        # Update dp and parent
        dp[j, j+1:m] = new_lengths
        parent[j, j+1:m] = preds
        
        # Update best
        # We need to check if any new_lengths is greater than best_len
        # new_lengths is a 1D array
        if new_lengths.size > 0:
            local_max = np.max(new_lengths)
            if local_max > best_len:
                best_len = local_max
                # Find the index i in j+1:m that achieved this
                # new_lengths corresponds to indices j+1, j+2, ... m-1
                local_idx = np.argmax(new_lengths)
                best_i = j
                best_j = j + 1 + local_idx
    
    # Reconstruct
    if best_len <= 2:
        return [primes[0], primes[1]]
        
    chain = [primes[best_j], primes[best_i]]
    ci, cj = best_i, best_j
    clen = best_len
    
    while clen > 2:
        k = parent[ci][cj]
        if k < 0:
            break
        chain.append(primes[k])
        cj = ci
        ci = k
        clen -= 1
        
    chain.reverse()
    return chain

