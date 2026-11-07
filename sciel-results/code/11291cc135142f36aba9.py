"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 70  # Leave 20s safety margin before 90s kill

    # Precompute smallest prime factor for numbers up to max candidate + 1
    max_candidate = max(hi for lo, hi in P['gaps'])
    spf = list(range(max_candidate + 1))
    for i in range(2, isqrt(max_candidate) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_candidate + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def least_prime_factor(m):
        """Return the least prime factor of m using precomputed SPF table."""
        if m <= max_candidate:
            return spf[m]
        for d in range(2, isqrt(m) + 1):
            if m % d == 0:
                return d
        return m

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # We need m in (lo, hi) such that least_prime_factor(m) >= gap
        # Try candidates in the middle of the gap first, then expand outward
        mid = (lo + hi) // 2
        
        # Generate candidates in order of distance from mid
        candidates = []
        for offset in range(0, (hi - lo) // 2 + 1):
            c1 = mid + offset
            c2 = mid - offset
            if lo < c1 < hi:
                candidates.append(c1)
            if lo < c2 < hi and c2 != c1:
                candidates.append(c2)

        found = False
        for m in candidates:
            if time.time() > deadline:
                break
            lpf = least_prime_factor(m)
            if lpf >= gap:
                result[i] = int(m)
                found = True
                break
        if found:
            continue
        # If not found via middle-out, try a more exhaustive scan with early termination
        # This handles cases where the middle-out heuristic might miss a valid witness
        # due to the specific distribution of numbers with large least prime factors
        if time.time() > deadline:
            break
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            lpf = least_prime_factor(m)
            if lpf >= gap:
                result[i] = int(m)
                break

    # Ensure all entries are Python int or None
    return [int(x) if x is not None else None for x in result]
# EVOLVE-BLOCK-END
