"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def construct():
    deadline = time.time() + 85.0
    from math import isqrt
    
    # Generate all primes <= 10000 using sieve
    sieve = [True] * 10001
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(10000) + 1):
        if sieve[i]:
            for j in range(i*i, 10001, i):
                sieve[j] = False
    primes = [p for p in range(2, 10001) if sieve[p]]
    n = len(primes)
    
    # DP: dp[i] = length of longest valid subsequence ending at primes[i]
    # prev_gap[i] = the gap used to end that subsequence (primes[i] - primes[j])
    # parent[i] = index j such that primes[j] -> primes[i] is the last step
    dp = [1] * n
    prev_gap = [0] * n
    parent = [-1] * n
    
    # For each prime i, find best j < i such that gap(i,j) >= prev_gap[j]
    # We want to maximize dp[j] where gap = primes[i] - primes[j] >= prev_gap[j]
    # This is a 2D optimization; we use a different approach:
    # Group by the last gap value. For each possible last gap g, keep the best dp.
    
    # Alternative: iterate and for each i, scan backwards but with early termination
    # Since n ~ 1229, O(n^2) is ~1.5M ops, which is fast enough.
    
    for i in range(1, n):
        pi = primes[i]
        best_len = 1
        best_parent = -1
        best_gap = 0
        for j in range(i):
            gap = pi - primes[j]
            if gap >= prev_gap[j]:
                if dp[j] + 1 > best_len or (dp[j] + 1 == best_len and gap < best_gap):
                    best_len = dp[j] + 1
                    best_parent = j
                    best_gap = gap
        dp[i] = best_len
        parent[i] = best_parent
        prev_gap[i] = best_gap
    
    # Find the index with maximum dp value
    max_idx = max(range(n), key=lambda i: dp[i])
    
    # Reconstruct
    result = []
    idx = max_idx
    while idx != -1:
        result.append(primes[idx])
        idx = parent[idx]
    result.reverse()
    
    return result
# EVOLVE-BLOCK-END
