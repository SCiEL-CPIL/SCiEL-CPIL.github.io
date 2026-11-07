"""Rewritten solver: find least-prime-factor witnesses inside consecutive prime gaps.

A witness m in (lo, hi) for gap g = hi - lo must have least prime factor >= g.
Since m is strictly between two consecutive primes, m is composite, so it has a
least prime factor; the condition is equivalent to: m is NOT divisible by any
prime p < g. The set of such forbidden primes depends only on g, and across all
32 pinned pairs the union of forbidden primes is fixed (primes < max_gap), so we
precompute it once and test each candidate with a handful of mod operations.
"""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_below(n):
    """Return the list of primes strictly less than n (n is small here)."""
    if n <= 2:
        return []
    sieve = [True] * n
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n:step] = [False] * len(sieve[start:n:step])
    return [i for i in range(2, n) if sieve[i]]

def construct():
    # Deadline guard: return best result safely before the 90 s wall-clock limit.
    deadline = time.monotonic() + 85.0

    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    max_gap = max(hi - lo for lo, hi in gaps)
    # Forbidden primes: any prime p < g. Union across all pairs = primes < max_gap.
    forbidden = _primes_below(max_gap)

    for i in range(n):
        lo, hi = gaps[i]
        g = hi - lo
        # Candidates strictly between lo and hi.
        for m in range(lo + 1, hi):
            # m is composite (between consecutive primes); check no forbidden prime divides it.
            ok = True
            for p in forbidden:
                if p >= g:
                    break
                if m % p == 0:
                    ok = False
                    break
            if ok:
                result[i] = m
                break
        if time.monotonic() > deadline:
            break

    # Ensure types are plain Python ints (they already are) and result is a list.
    return result
# EVOLVE-BLOCK-END
