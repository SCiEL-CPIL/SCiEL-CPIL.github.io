"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def construct():
    """
    Find the longest increasing subsequence of primes <= 10000
    with nondecreasing successive gaps.
    
    Uses dynamic programming: dp[i] = length of longest valid subsequence ending at prime[i]
    """
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    
    # Generate primes <= 10000 using Sieve of Eratosthenes
    limit = 10000
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.array(np.where(sieve)[0])
    n = len(primes)
    
    if n < 2:
        return [int(p) for p in primes]
    
    # DP: dp[i] = max length of valid subsequence ending at index i
    # A valid subsequence has nondecreasing gaps
    # dp[i] = max over j < i where primes[i] - primes[j] >= (primes[j] - primes[j-1]) if j > 0
    #         of dp[j] + 1
    # For j == 0, any gap is valid (no previous gap to compare)
    
    dp = np.ones(n, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    
    # Process in order
    for i in range(1, n):
        if time.time() > deadline:
            break
        best_len = 1
        best_j = -1
        pi = primes[i]
        
        for j in range(i):
            gap = pi - primes[j]
            if j == 0:
                # No previous gap constraint
                if dp[j] + 1 > best_len:
                    best_len = dp[j] + 1
                    best_j = j
            else:
                prev_gap = primes[j] - primes[j-1]
                if gap >= prev_gap:
                    if dp[j] + 1 > best_len:
                        best_len = dp[j] + 1
                        best_j = j
        
        dp[i] = best_len
        parent[i] = best_j
    
    # Find the index with maximum dp value
    max_idx = int(np.argmax(dp))
    max_len = int(dp[max_idx])
    
    # Reconstruct the subsequence
    result = []
    idx = max_idx
    while idx != -1:
        result.append(int(primes[idx]))
        idx = int(parent[idx])
    result.reverse()
    
    return result
# EVOLVE-BLOCK-END
