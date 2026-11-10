import time
from math import isqrt
from bisect import bisect_right

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    # 1. Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N < 2:
        return primes
        
    # dp[i][j] stores the length of the longest valid sequence ending with primes[j], primes[i] (j < i)
    # prev[i][j] stores the index k such that primes[k], primes[j] was the previous pair
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    max_len = 2
    max_i, max_j = 0, 1
    
    # Iterate over the second-to-last prime index j
    for j in range(N):
        # Prepare data for predecessors k < j
        candidates = []
        for k in range(j):
            if dp[j][k] > 0:
                gap = primes[j] - primes[k]
                candidates.append((gap, dp[j][k], k))
        
        if not candidates:
            # No predecessors, so for all i > j, dp[i][j] = 2
            for i in range(j + 1, N):
                dp[i][j] = 2
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    max_i, max_j = i, j
            continue
            
        # Sort by gap
        candidates.sort(key=lambda x: x[0])
        
        # Create arrays for binary search
        gaps = [c[0] for c in candidates]
        lengths = [c[1] for c in candidates]
        ks = [c[2] for c in candidates]
        
        # Compute prefix max
        prefix_max_len = [0] * len(lengths)
        prefix_max_idx = [0] * len(lengths)
        current_max = 0
        current_idx = 0
        for idx in range(len(lengths)):
            if lengths[idx] > current_max:
                current_max = lengths[idx]
                current_idx = idx
            prefix_max_len[idx] = current_max
            prefix_max_idx[idx] = current_idx
        
        # For each i > j, find the best k
        for i in range(j + 1, N):
            if time.time() > DEADLINE:
                break
                
            target_gap = primes[i] - primes[j]
            
            # Find rightmost index in gaps where gap <= target_gap
            pos = bisect_right(gaps, target_gap)
            
            if pos > 0:
                best_idx_in_candidates = prefix_max_idx[pos - 1]
                best_len_val = prefix_max_len[pos - 1]
                best_k = ks[best_idx_in_candidates]
                
                dp[i][j] = best_len_val + 1
                prev[i][j] = best_k
                
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    max_i, max_j = i, j
            else:
                # No valid predecessor k
                dp[i][j] = 2
                prev[i][j] = -1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    max_i, max_j = i, j
                    
        if time.time() > DEADLINE:
            break

    # Reconstruct the sequence
    if max_len < 2:
        return primes[:max_len]
        
    path = []
    i, j = max_i, max_j
    path.append(i)
    path.append(j)
    
    while prev[i][j] != -1:
        k = prev[i][j]
        i, j = j, k
        path.append(j)
        
    path.reverse()
    
    result_primes = [primes[idx] for idx in path]
    
    return result_primes
