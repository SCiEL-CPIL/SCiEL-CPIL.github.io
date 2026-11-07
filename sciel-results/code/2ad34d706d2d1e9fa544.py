"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    max_gap = max(hi - lo for lo, hi in P['gaps'])

    # Sieve for primes up to max_gap
    if max_gap >= 2:
        sieve = [True] * (max_gap + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_gap**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, max_gap + 1, i):
                    sieve[j] = False
        primes_up_to_max_gap = [i for i in range(2, max_gap + 1) if sieve[i]]
    else:
        primes_up_to_max_gap = []

    # Precompute forbidden prime lists for each gap size
    # For gap g, forbidden primes are all primes < g
    forbidden_primes_by_gap = {}
    for g in range(2, max_gap + 1):
        fp = [p for p in primes_up_to_max_gap if p < g]
        forbidden_primes_by_gap[g] = fp

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        fp = forbidden_primes_by_gap[gap]
        
        # Scan candidates from lo+1 to hi-1
        # A candidate m is valid iff m is not divisible by any prime in fp
        found = False
        for m in range(lo + 1, hi):
            valid = True
            for p in fp:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                found = True
                break

    # Ensure all entries are Python int or None
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))

    return final_result
# EVOLVE-BLOCK-END
