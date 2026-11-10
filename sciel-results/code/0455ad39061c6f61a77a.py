"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _build_spf(limit):
    """Build smallest-prime-factor sieve up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _powerful_part(n, spf):
    """Return the powerful part of n: product of p^e for all prime factors with e>=2."""
    if n <= 1:
        return 1
    result = 1
    while n > 1:
        p = spf[n]
        cnt = 0
        while n % p == 0:
            n //= p
            cnt += 1
        if cnt >= 2:
            result *= p ** cnt
    return result

def construct():
    deadline = time.time() + 80.0  # 80s safety margin before 90s kill
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # need to factor up to 1000000, and n+1 up to 1000001

    # Build SPF table
    spf = _build_spf(limit)

    # Full scan: compute powerful part for each n in [10000, 1000000]
    # and track the max of powerful_part(n) * powerful_part(n+1)
    best_val = 0
    best_n = N_MIN

    # Precompute powerful parts for all numbers in range to avoid redundant work
    # We need pp(n) for n in [10000, 1000001]
    # Compute pp for all numbers from 1 to 1000001
    max_n = N_MAX + 1  # 1000001
    pp = [1] * (max_n + 1)

    for i in range(2, max_n + 1):
        if time.time() > deadline:
            break
        # Factor i using spf
        n = i
        result = 1
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt >= 2:
                result *= p ** cnt
        pp[i] = result

    # Now scan consecutive pairs
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n
# EVOLVE-BLOCK-END
