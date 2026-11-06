import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # 5s margin from 90s limit, though this should be much faster
    
    # 1. Generate primes up to 10000
    limit_val = 10000
    is_prime = [True] * (limit_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_val + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # 2. DP Setup
    # dp[e][p] = length of longest valid subsequence ending at primes[e] with previous prime primes[p]
    # Using numpy arrays for better performance
    dp = np.zeros((N, N), dtype=np.int32)
    # choice[e][p] = index k such that dp[p][k] was the max used to compute dp[e][p]
    # If no such k, choice is -1
    choice = np.full((N, N), -1, dtype=np.int32)
    
    # For each p, we maintain a pointer to the next k to consider
    # and the current max value for the range [k_ptr+1, p-1]
    k_ptr = np.array([p - 1 for p in range(N)], dtype=np.int32)
    max_val = np.zeros(N, dtype=np.int32)
    # argmax_val[p] stores the index k that achieved max_val[p]
    argmax_idx = np.full(N, -1, dtype=np.int32)
    
    best_len = 2
    best_e = 1
    best_p = 0
    
    # 3. DP Iteration
    for e in range(N):
        if time.time() - start_time > deadline:
            break
        for p in range(e):
            gap = primes[e] - primes[p]
            limit = primes[p] - gap
            
            # Update max_val[p] with new k's that become valid
            # k must satisfy primes[k] >= limit
            # We add k's in descending order of index (which is descending order of value)
            # k_ptr starts at p-1. We decrement it.
            while k_ptr[p] >= 0 and primes[k_ptr[p]] >= limit:
                k = k_ptr[p]
                if dp[p, k] > max_val[p]:
                    max_val[p] = dp[p, k]
                    argmax_idx[p] = k
                k_ptr[p] -= 1
            
            if max_val[p] > 0:
                dp[e, p] = max_val[p] + 1
                choice[e, p] = argmax_idx[p]
            else:
                dp[e, p] = 2
                choice[e, p] = -1
            
            if dp[e, p] > best_len:
                best_len = dp[e, p]
                best_e = e
                best_p = p
                
    # 4. Reconstruct Sequence
    seq = []
    if best_len >= 2:
        seq.append(primes[best_e])
        seq.append(primes[best_p])
        curr_e, curr_p = best_e, best_p
        k = choice[curr_e, curr_p]
        while k != -1:
            seq.append(primes[k])
            curr_e, curr_p = curr_p, k
            k = choice[curr_e, curr_p]
        
    seq.reverse()
    return seq
