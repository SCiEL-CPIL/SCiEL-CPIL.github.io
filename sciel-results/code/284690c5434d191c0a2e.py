import time
import math
import numpy as np

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = np.array([i for i, v in enumerate(is_prime) if v], dtype=np.int64)
    N = len(primes)
    
    if N <= 1:
        return primes.tolist()
    
    # dp[i][j] = max length of valid subsequence ending at primes[i], primes[j] (i < j)
    # Initialize all pairs to length 2
    dp = np.full((N, N), 2, dtype=np.int32)
    parent = np.full((N, N), -1, dtype=np.int32)
    
    # suffix_max[i][k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i]) for k in range(i)
    # We'll compute suffix_max for each i as needed
    # suffix_max[i] is an array of length i
    # suffix_max_parent[i][k] = argmax index
    
    best_len = 2
    best_end = (0, 1)
    
    for j in range(1, N):
        if time.time() > deadline:
            break
        
        # For each i < j, compute dp[i][j]
        # We need suffix_max[i] which depends on dp[k][i] for k < i
        # dp[k][i] are finalized after processing j=i
        
        # Precompute suffix_max for all i in range(j) if not already done
        # Actually, we can compute it on the fly for each i
        
        for i in range(j):
            if time.time() > deadline:
                break
            
            # threshold = 2 * primes[i] - primes[j]
            # We need to find smallest k < i such that primes[k] >= threshold
            # If such k exists, dp[i][j] = 1 + max(dp[k][i] for k in [k_min, i-1])
            # Which is 1 + suffix_max[i][k_min]
            
            threshold = 2 * primes[i] - primes[j]
            
            # Binary search in primes[0:i] for threshold
            # Find smallest k such that primes[k] >= threshold
            lo, hi = 0, i
            while lo < hi:
                mid = (lo + hi) // 2
                if primes[mid] >= threshold:
                    hi = mid
                else:
                    lo = mid + 1
            k_min = lo
            
            if k_min < i:
                # Need suffix_max[i][k_min]
                # suffix_max[i][k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i])
                # We need the max of dp[k_min:i, i]
                # i.e., max(dp[k][i] for k in range(k_min, i))
                
                # Compute this max
                if k_min == i - 1:
                    best_prev = dp[k_min, i]
                    best_parent = k_min
                else:
                    # Find max in dp[k_min:i, i]
                    segment = dp[k_min:i, i]
                    max_idx = np.argmax(segment)
                    best_prev = segment[max_idx]
                    best_parent = k_min + max_idx
                
                val = 1 + best_prev
                if val > dp[i, j]:
                    dp[i, j] = val
                    parent[i, j] = best_parent
                
                if dp[i, j] > best_len:
                    best_len = dp[i, j]
                    best_end = (i, j)
    
    # Reconstruct
    if best_len <= 1:
        return primes[:best_len].tolist()
    
    result = []
    ci, cj = best_end
    result.append(primes[cj])
    result.append(primes[ci])
    
    while parent[ci, cj] != -1:
        ck = parent[ci, cj]
        result.append(primes[ck])
        cj = ci
        ci = ck
    
    result.reverse()
    return result
