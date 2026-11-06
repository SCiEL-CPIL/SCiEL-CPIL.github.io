import time
import math
import numpy as np

def construct():
    deadline = time.time() + 85
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)
    
    if N <= 1:
        return primes
    
    # dp[i][j] = max length of valid subsequence ending at primes[i], primes[j] (i < j)
    # parent[i][j] = k such that dp[i][j] = dp[k][i] + 1
    dp = [[2] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]
    
    best_len = 2
    best_end = (0, 1) if N > 1 else (0, 0)
    
    # Loop i from 0 to N-2
    # For each i, we compute dp[i][j] for j > i
    for i in range(N - 1):
        # Precompute suffix max of dp[k][i] for k < i
        # suf_val[k] = max(dp[m][i] for m in [k, i-1])
        # suf_idx[k] = argmax
        if i > 0:
            suf_val = [0] * i
            suf_idx = [-1] * i
            suf_val[i-1] = dp[i-1][i]
            suf_idx[i-1] = i-1
            for k in range(i-2, -1, -1):
                if dp[k][i] > suf_val[k+1]:
                    suf_val[k] = dp[k][i]
                    suf_idx[k] = k
                else:
                    suf_val[k] = suf_val[k+1]
                    suf_idx[k] = suf_idx[k+1]
        else:
            suf_val = []
            suf_idx = []
            
        p_i = primes[i]
        # k_ptr is the smallest k < i such that primes[k] + p_i >= 2 * primes[j]
        # As j increases, 2*primes[j] increases, so the condition becomes stricter,
        # meaning valid k's must be larger. So k_ptr is non-decreasing.
        k_ptr = 0
        
        for j in range(i + 1, N):
            # Update k_ptr
            # Condition: primes[k] + p_i >= 2 * primes[j]
            # We want the smallest k satisfying this.
            # If current k_ptr does not satisfy it, we must increase k_ptr.
            while k_ptr < i and primes[k_ptr] + p_i < 2 * primes[j]:
                k_ptr += 1
            
            if k_ptr < i:
                # Valid k exists in [k_ptr, i-1]
                # The best is at k_ptr because suf_val is max over [k, i-1]
                # Wait, suf_val[k] is max over [k, i-1].
                # So suf_val[k_ptr] is the max over all valid k's.
                new_len = 1 + suf_val[k_ptr]
                dp[i][j] = new_len
                parent[i][j] = suf_idx[k_ptr]
                
                if new_len > best_len:
                    best_len = new_len
                    best_end = (i, j)
            else:
                # No valid k, base case
                dp[i][j] = 2
                parent[i][j] = -1
                # best_len is at least 2, so no update needed unless best_len < 2
                # which is not possible.

        # Check time periodically
        if i % 100 == 0 and time.time() > deadline:
            break

    # Reconstruct the sequence
    result = []
    ci, cj = best_end
    result.append(primes[cj])
    result.append(primes[ci])
    
    while parent[ci][cj] != -1:
        ck = parent[ci][cj]
        result.append(primes[ck])
        cj = ci
        ci = ck
        
    result.reverse()
    return result
