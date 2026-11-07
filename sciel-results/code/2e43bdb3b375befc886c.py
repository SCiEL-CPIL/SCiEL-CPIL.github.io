"""Rewritten program: Branch-and-bound over primes for max pairwise-coprime sum."""

def construct():
    import math
    import time

    DEADLINE = time.time() + 80  # 80s wall-clock safety margin (limit is 90s)
    N = 256

    # Sieve primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, N + 1) if is_prime[p]]

    # Sort primes descending for better pruning in DFS
    primes_desc = sorted(primes, reverse=True)

    # Precompute suffix sums for upper bound
    # suffix_sum[i] = sum of primes_desc[i:]
    suffix_sum = [0] * (len(primes_desc) + 1)
    for i in range(len(primes_desc) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + primes_desc[i]

    # Each prime can be represented as a single bit (since primes are pairwise coprime)
    # But we also need to consider composites. However, including any composite p*q
    # would force exclusion of both p and q, and p*q < p + q for p,q >= 2.
    # So composites are never beneficial. We only need to consider primes and 1.

    # The problem reduces to: select a subset of primes (each pairwise coprime trivially)
    # plus optionally 1, to maximize the sum.
    # Since all primes are pairwise coprime, we can include ALL primes plus 1.
    # Wait - that means the answer is simply 1 + sum of all primes <= 256.

    # Let me verify: are all primes pairwise coprime? Yes, by definition.
    # So the optimal solution is {1} ∪ {all primes in 2..256}.

    result = [1] + primes_desc  # include 1 and all primes
    return result
