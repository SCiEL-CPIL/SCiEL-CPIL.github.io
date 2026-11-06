import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 80  # safety margin from 90s limit

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

    # 2. DP: dp[p] = length of longest valid subsequence ending at primes[p]
    # last_gap[p] = the gap between primes[p] and its predecessor in the optimal subsequence ending at p
    # For p=0: dp=1, last_gap=0, prev=-1
    # For p>0: dp[p] = max(2, max over k<p where (primes[p]-primes[k]) >= last_gap[k] of dp[k]+1)
    #          last_gap[p] = primes[p] - primes[best_k]
    #          prev[p] = best_k
    
    dp = np.ones(N, dtype=np.int32)
    last_gap = np.zeros(N, dtype=np.int32)
    prev = np.full(N, -1, dtype=np.int32)
    
    best_len = 1
    best_p = 0
    
    for p in range(1, N):
        if time.time() - start_time > deadline:
            break
        best_k = -1
        best_val = 0
        for k in range(p):
            if last_gap[k] <= primes[p] - primes[k]:
                if dp[k] > best_val:
                    best_val = dp[k]
                    best_k = k
        if best_k >= 0:
            dp[p] = best_val + 1
            last_gap[p] = primes[p] - primes[best_k]
            prev[p] = best_k
        else:
            dp[p] = 2
            last_gap[p] = 0
            prev[p] = -1
        
        if dp[p] > best_len:
            best_len = dp[p]
            best_p = p

    # 3. Reconstruct
    seq = []
    curr = best_p
    while curr != -1:
        seq.append(primes[curr])
        curr = prev[curr]
    seq.reverse()
    
    return seq
