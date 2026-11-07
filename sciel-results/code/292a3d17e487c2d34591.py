"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75

    def sieve(n):
        is_p = bytearray(b'\x01') * (n + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_p[i]:
                for j in range(i * i, n + 1, i):
                    is_p[j] = 0
        return [i for i in range(2, n + 1) if is_p[i]]

    # Sieve up to max gap to get all primes we need to check
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    primes_small = sieve(max_gap + 1)

    # Precompute for each distinct gap size the list of primes < gap
    gap_to_primes = {}
    for (lo, hi) in P['gaps']:
        g = hi - lo
        if g not in gap_to_primes:
            gap_to_primes[g] = [p for p in primes_small if p < g]

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        if gap <= 1:
            continue

        check_primes = gap_to_primes[gap]

        if len(check_primes) == 0:
            result[i] = int(lo + 1)
            continue

        # Use numpy for faster marking
        size = gap - 1  # candidates: lo+1, ..., hi-1
        bad = np.zeros(size, dtype=np.uint8)

        for p in check_primes:
            rem = (lo + 1) % p
            if rem == 0:
                first = lo + 1
            else:
                first = lo + 1 + (p - rem)
            if first >= hi:
                continue
            start_offset = first - (lo + 1)
            bad[start_offset::p] = 1

        # Find good positions
        good_indices = np.where(bad == 0)[0]

        if len(good_indices) > 0:
            result[i] = int(lo + 1 + good_indices[0])
        else:
            # Fallback: brute-force verify each candidate
            for m in range(lo + 1, hi):
                valid = True
                for p in check_primes:
                    if m % p == 0:
                        valid = False
                        break
                if valid:
                    result[i] = int(m)
                    break

    # Final verification pass to ensure correctness
    for i, (lo, hi) in enumerate(P['gaps']):
        w = result[i]
        if w is None:
            continue
        gap = hi - lo
        if not (lo < w < hi):
            result[i] = None
            continue
        # Verify LPF(w) >= gap
        for p in gap_to_primes[gap]:
            if w % p == 0:
                result[i] = None
                break

    return result
# EVOLVE-BLOCK-END
