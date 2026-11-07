"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_primes(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]


def _least_prime_factor(m, small_primes):
    """
    Return the least prime factor of m.
    If m is prime, return m itself.
    small_primes: list of primes to check first.
    """
    if m < 2:
        return m
    for p in small_primes:
        if p * p > m:
            return m  # m is prime (or its least factor is itself)
        if m % p == 0:
            return p
    # If no small prime divides m and m is not too large, m is prime
    # For our range (~10000), checking up to sqrt(m) ~ 100 is sufficient
    # But small_primes should cover up to sqrt(max_m)
    # If we get here, m has no factor in small_primes up to sqrt(m)
    # So m is prime
    return m


def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, leaving 10s safety margin

    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    # Precompute primes up to a reasonable limit
    # Max gap is 10289 - 10273 = 16, but we need primes up to sqrt(max_m) ~ sqrt(10289) ~ 102
    # Also need primes up to max_gap for the condition
    max_m = max(hi for _, hi in gaps)
    max_gap = max(hi - lo for lo, hi in gaps)
    # We need primes up to max(max_gap, isqrt(max_m))
    import math
    sieve_limit = max(max_gap, math.isqrt(max_m) + 10, 200)
    small_primes = _sieve_primes(sieve_limit)

    # For each gap, search for a witness
    for i, (lo, hi) in enumerate(gaps):
        gap = hi - lo
        # Search in the middle of the gap first, then expand outward
        mid = (lo + hi) // 2
        # Generate candidates in order of distance from mid
        candidates = []
        for d in range(0, (hi - lo) // 2 + 1):
            c1 = mid + d
            c2 = mid - d
            if lo < c1 < hi:
                candidates.append(c1)
            if lo < c2 < hi and c2 != c1:
                candidates.append(c2)

        # Also try from lo+1 and hi-1 inward as a backup
        found = False
        for m in candidates:
            if time.time() > deadline:
                break
            lpf = _least_prime_factor(m, small_primes)
            if lpf >= gap:
                result[i] = m
                found = True
                break

        if not found:
            # Try a more systematic search from both ends
            for m in range(lo + 1, hi):
                if time.time() > deadline:
                    break
                lpf = _least_prime_factor(m, small_primes)
                if lpf >= gap:
                    result[i] = m
                    break

    # Ensure all entries are Python int or None
    for i in range(n):
        if result[i] is not None:
            result[i] = int(result[i])

    return result
# EVOLVE-BLOCK-END
