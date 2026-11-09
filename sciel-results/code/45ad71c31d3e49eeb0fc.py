"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    max_val = 0
    for p, q in P['gaps']:
        if q > max_val:
            max_val = q
    max_val += 10

    # Linear sieve for least prime factor
    lpf = [0] * (max_val + 1)
    primes = []
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            lpf[i] = i
            primes.append(i)
        for p in primes:
            if p > lpf[i] or i * p > max_val:
                break
            lpf[i * p] = p

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Search from both ends inward to find a witness quickly
        found = False
        # Try from lo+1 upward
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                found = True
                break
        if not found:
            # Also try from hi-1 downward (in case we missed something, though logic is same)
            for m in range(hi - 1, lo, -1):
                if lpf[m] >= gap:
                    result[i] = m
                    found = True
                    break
        if not found:
            result[i] = None

    return result
# EVOLVE-BLOCK-END
