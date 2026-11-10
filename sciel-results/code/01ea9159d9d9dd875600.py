"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def construct():
    """
    Return two positive integers a, b with a + b = 123457.
    Minimize max(P(a), P(b)) where P(n) is the largest prime factor of n (P(1)=1).
    """
    target = 123457
    deadline = time.time() + 75  # safety margin before 90s kill

    # Step 1: Compute smallest prime factor (SPF) up to target using a standard sieve.
    # This allows O(log n) factorization for any number <= target.
    spf = list(range(target + 1))
    for i in range(2, int(math.isqrt(target)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, target + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Step 2: Precompute largest prime factor for all numbers up to target.
    # Using a dynamic programming approach:
    # lp[n] = max(spf[n], lp[n // spf[n]])
    # This is O(N) overall and much faster than repeated trial division.
    lp = [1] * (target + 1)
    for n in range(2, target + 1):
        p = spf[n]
        rest = n // p
        lp[n] = max(p, lp[rest])

    # Step 3: For each a in [1, target-1], b = target - a
    # Compute max(lp[a], lp[b]) and find the minimum.
    # We iterate a from 1 to target//2 due to symmetry.
    best_val = float('inf')
    best_a = 1
    best_b = target - 1

    for a in range(1, target // 2 + 1):
        b = target - a
        val = max(lp[a], lp[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
        if time.time() > deadline:
            break

    return [best_a, best_b]

# EVOLVE-BLOCK-END
