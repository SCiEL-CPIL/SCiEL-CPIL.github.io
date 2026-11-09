"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_up_to(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

# Precompute least prime factor (LPF) up to max prime in gaps
_MAX_PRIME = max(hi for _, hi in P['gaps'])
_LPF_SIZE = _MAX_PRIME + 1

def _build_lpf(n):
    """Build least prime factor array up to n using a modified sieve."""
    lpf = list(range(n + 1))
    lpf[0] = 0
    if n >= 1:
        lpf[1] = 1
    for i in range(2, isqrt(n) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i*i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf

_LPF = _build_lpf(_MAX_PRIME)

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin from 90s limit
    
    n = len(P['gaps'])
    result = [None] * n
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - start_time > DEADLINE:
            break
        gap = hi - lo
        # Find a valid witness: m with least prime factor >= gap
        # LPF[m] gives the least prime factor of m
        # We need LPF[m] >= gap (or m is prime, in which case LPF[m] == m)
        found = False
        for m in range(lo + 1, hi):
            if _LPF[m] >= gap:
                result[i] = m
                found = True
                break
        if not found:
            # Try reverse scan as safety
            for m in range(hi - 1, lo, -1):
                if _LPF[m] >= gap:
                    result[i] = m
                    found = True
                    break
    
    # Ensure all are Python int or None
    return [int(x) if x is not None else None for x in result]
# EVOLVE-BLOCK-END
