"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_primes_up_to(n):
    """Return a list of all primes <= n (n >= 2)."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start::step] = bytearray(len(sieve[start::step]))
    return [i for i in range(2, n + 1) if sieve[i]]


def _compute_lpf_array(n):
    """Compute least prime factor for all numbers up to n.
    Returns array where lpf[i] = least prime factor of i, or i if i is prime, 1 for i=1, 0 for i=0."""
    lpf = list(range(n + 1))  # lpf[i] = i initially (assume prime)
    lpf[0] = 0
    if n >= 1:
        lpf[1] = 1
    for i in range(2, isqrt(n) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:  # not yet assigned
                    lpf[j] = i
    return lpf


def construct():
    deadline = time.time() + 55.0  # safety margin before 60s limit

    # Determine the range of numbers we need to check
    min_lo = min(lo for lo, hi in P['gaps'])
    max_hi = max(hi for lo, hi in P['gaps'])
    
    # Max gap determines how many primes we need to test
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # We need lpf for all numbers in [min_lo+1, max_hi-1]
    # But lpf computation is easier if we do it from 0 to max_hi
    # For efficiency, compute lpf up to max_hi
    lpf = _compute_lpf_array(max_hi)
    
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find m in (lo, hi) such that lpf[m] >= gap
        # lpf[m] is the least prime factor; if m is prime, lpf[m] = m
        # We need lpf[m] >= gap
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = int(m)
                break
    
    # Return a plain list of Python ints / None, exactly 32 entries
    return [int(x) if x is not None else None for x in result]
# EVOLVE-BLOCK-END
