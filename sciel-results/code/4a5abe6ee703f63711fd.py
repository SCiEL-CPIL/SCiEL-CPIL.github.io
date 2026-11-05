"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    import time
    
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    # 1. Generate all primes <= 10000 using Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes

    # 2. Dynamic Programming to find the longest subsequence with non-decreasing gaps
    # dp[i] = length of longest valid subsequence ending at primes[i]
    # prev[i] = index of the previous prime in that subsequence (for reconstruction)
    
    dp = [1] * n
    prev = [-1] * n
    max_len = 1
    max_idx = 0

    # Iterate through primes. For each prime j, look back at all i < j.
    # If gap(j) >= gap(i), then we can extend the subsequence ending at i to j.
    # gap(j) relative to i is primes[j] - primes[i].
    # The previous gap in the subsequence ending at i is (primes[i] - primes[prev[i]]) if prev[i] != -1.
    # Condition: primes[j] - primes[i] >= (primes[i] - primes[prev[i]])
    
    for j in range(1, n):
        if time.time() > DEADLINE:
            break # Should not happen for n~1200, but good practice
            
        for i in range(j):
            gap_new = primes[j] - primes[i]
            
            if prev[i] == -1:
                # Subsequence ending at i has length 1, no previous gap constraint
                # Any gap is "non-decreasing" compared to nothing
                if dp[i] + 1 > dp[j]:
                    dp[j] = dp[i] + 1
                    prev[j] = i
            else:
                gap_prev = primes[i] - primes[prev[i]]
                if gap_new >= gap_prev:
                    if dp[i] + 1 > dp[j]:
                        dp[j] = dp[i] + 1
                        prev[j] = i
        
        if dp[j] > max_len:
            max_len = dp[j]
            max_idx = j

    # 3. Reconstruct the subsequence
    result = []
    idx = max_idx
    while idx != -1:
        result.append(primes[idx])
        idx = prev[idx]
    
    result.reverse()
    return result
# EVOLVE-BLOCK-END
