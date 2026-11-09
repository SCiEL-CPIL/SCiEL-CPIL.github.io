"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _build_lpf(n):
    """Build least prime factor table up to n using linear sieve."""
    lpf = [0] * (n + 1)
    primes = []
    for i in range(2, n + 1):
        if lpf[i] == 0:
            lpf[i] = i
            primes.append(i)
        for p in primes:
            if p > lpf[i] or i * p > n:
                break
            lpf[i * p] = p
    return lpf


def construct():
    deadline = time.time() + 55  # safety margin before 60s kill
    max_prime = max(hi for lo, hi in P['gaps'])
    lpf = _build_lpf(max_prime)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find any m in (lo, hi) with lpf[m] >= gap
        # lpf[m] == m means m is prime, which satisfies lpf[m] >= gap when m >= gap
        # For composites, lpf[m] is the smallest prime factor
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break
    
    return result
# EVOLVE-BLOCK-END
