"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Step 1: Generate all primes <= 10000 using Sieve of Eratosthenes
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve_primes(10000)
    n = len(primes)

    # Step 2: DP to find the longest non-decreasing-gap prime subsequence
    # dp[i] = length of longest valid subsequence ending at primes[i]
    # prev[i] = index of previous element in that subsequence
    # gap[i] = gap between primes[i] and the element before it in the subsequence
    #
    # Transition: for j < i, we can append primes[i] after primes[j] if
    #   primes[i] - primes[j] >= gap[j]
    # where gap[j] is the gap in the subsequence ending at j (or 0 if j is the first element)
    #
    # We want to maximize length. Among ties, we want to keep track of the actual sequence.

    dp = [1] * n
    prev_idx = [-1] * n
    gap = [0] * n  # gap[j] = primes[j] - primes[prev[j]] if prev[j] >= 0, else 0

    for i in range(n):
        best_len = 1
        best_prev = -1
        best_gap = 0
        for j in range(i):
            # Can we extend the subsequence ending at j with primes[i]?
            # Need: primes[i] - primes[j] >= gap[j]
            if primes[i] - primes[j] >= gap[j]:
                if dp[j] + 1 > best_len:
                    best_len = dp[j] + 1
                    best_prev = j
                    best_gap = primes[i] - primes[j]
        dp[i] = best_len
        prev_idx[i] = best_prev
        gap[i] = best_gap

    # Check time
    if time.time() > DEADLINE:
        # Fallback: return greedy result
        values = [primes[0]]
        for p in primes[1:]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

    # Step 3: Find the index with maximum dp value
    max_len = max(dp)
    end_idx = dp.index(max_len)

    # Step 4: Reconstruct the sequence
    result = []
    idx = end_idx
    while idx >= 0:
        result.append(primes[idx])
        idx = prev_idx[idx]
    result.reverse()

    return result
# EVOLVE-BLOCK-END
