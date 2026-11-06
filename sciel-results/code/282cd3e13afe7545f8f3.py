import time
import math
from bisect import bisect_left

def construct():
    deadline = time.time() + 78  # 78s deadline, 12s safety margin
    
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
    # We store dp as a 2D list, dp[i][j] for i < j
    dp = [[2] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]
    
    # suffix_max[i] = array where suffix_max[i][k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i])
    # suffix_max_parent[i][k] = the index that achieves the max
    suffix_max = [None] * N
    suffix_max_parent = [None] * N
    
    best_len = 2
    best_end = (0, 1)
    
    for j in range(1, N):
        # Compute suffix_max for all i < j that we haven't computed yet
        for i in range(j):
            if suffix_max[i] is None:
                sm = [0] * i
                sm_parent = [-1] * i
                if i > 0:
                    sm[i-1] = dp[i-1][i]
                    sm_parent[i-1] = i-1
                    for k in range(i-2, -1, -1):
                        if dp[k][i] > sm[k+1]:
                            sm[k] = dp[k][i]
                            sm_parent[k] = k
                        else:
                            sm[k] = sm[k+1]
                            sm_parent[k] = sm_parent[k+1]
                suffix_max[i] = sm
                suffix_max_parent[i] = sm_parent
        
        # Now compute dp[i][j] for i in range(j)
        for i in range(j):
            if time.time() > deadline:
                break
            threshold = 2 * primes[i] - primes[j]
            # Find smallest k such that primes[k] >= threshold in primes[0:i]
            k_min = bisect_left(primes, threshold, 0, i)
            
            if k_min < i:
                val = 1 + suffix_max[i][k_min]
                if val > dp[i][j]:
                    dp[i][j] = val
                    parent[i][j] = suffix_max_parent[i][k_min]
            
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_end = (i, j)
        if time.time() > deadline:
            break
    
    # Reconstruct the sequence via backtracking
    if best_len <= 1:
        return primes[:best_len]
    
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
