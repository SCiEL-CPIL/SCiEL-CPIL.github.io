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


def construct():
    # Explicit wall-clock deadline guard (evaluator kills after ~90s; use a margin).
    deadline = time.time() + 60.0

    # Max possible gap among the pinned pairs -> bound on any prime factor we test.
    max_gap = max(hi - lo for lo, hi in P['gaps'])

    # Primes up to a safe bound. lpf(m) >= gap only needs testing primes < gap,
    # but we sieve generously so the per-candidate check is a simple list scan.
    primes = _sieve_primes_up_to(max(max_gap, 100))

    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Candidate m is a valid witness iff it has no prime factor p < gap,
        # i.e. its least prime factor is >= gap (or m itself is prime >= gap).
        for m in range(lo + 1, hi):
            ok = True
            for p in primes:
                if p >= gap:
                    break  # remaining primes are >= gap; none can be a factor
                if m % p == 0:
                    ok = False
                    break
            if ok:
                result[i] = int(m)  # exact Python int (rejects bool/NumPy)
                break

    # Return a plain list of Python ints / None, exactly 32 entries.
    return [int(x) if x is not None else None for x in result]
# EVOLVE-BLOCK-END
