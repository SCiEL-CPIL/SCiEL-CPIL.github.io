"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _build_lpf_sieve(limit):
    """
    Linear sieve to compute least prime factor for all integers up to limit.
    Returns a list lpf where lpf[n] is the least prime factor of n (0 if n < 2).
    """
    lpf = [0] * (limit + 1)
    primes = []
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            lpf[i] = i
            primes.append(i)
        for p in primes:
            if p > lpf[i] or i * p > limit:
                break
            lpf[i * p] = p
    return lpf, primes

def construct():
    deadline = time.time() + 75.0
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    max_val = max(hi for _, hi in gaps)
    lpf, primes = _build_lpf_sieve(max_val)

    # Precompute for each gap: the set of primes strictly less than the gap size
    # Actually, we just need to check lpf[m] >= g for each candidate m
    # Since lpf[m] gives the least prime factor directly, this is O(1) per check

    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        g = hi - lo

        # Search from lo+1 to hi-1 for m where lpf[m] >= g
        # lpf[m] >= g means m has no prime factor less than g
        # Note: if m is prime, lpf[m] = m which is > g (since m < hi and hi-lo=g, m could be close to hi)
        # Actually m < hi, and g = hi - lo, so m can be up to hi-1
        # lpf[m] >= g is the condition

        found = False
        for m in range(lo + 1, hi):
            # lpf[m] is the least prime factor of m
            # We need lpf[m] >= g
            if lpf[m] >= g:
                result[i] = int(m)
                found = True
                break

    return result
# EVOLVE-BLOCK-END
