"""
Rewritten program using a precomputed Least Prime Factor (LPF) sieve.
This approach is O(N log log N) for precomputation and O(G) for scanning,
where G is the total size of the gaps. This is significantly faster and
more robust than trial division for each candidate.
"""

import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    """
    Find witnesses m in (lo, hi) such that lpf(m) >= gap = hi - lo.
    Uses a precomputed LPF array for O(1) lookup.
    """
    # Determine the maximum value we need to check
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Sieve for Least Prime Factor (LPF)
    # lpf[i] will store the smallest prime factor of i
    # For primes, lpf[p] = p
    # For composites, lpf[n] = smallest prime divisor
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, max_val + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
                    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        witness = None
        # Scan integers strictly between lo and hi
        for m in range(lo + 1, hi):
            # Check if least prime factor is at least the gap
            if lpf[m] >= gap:
                witness = m
                break
        result[i] = witness
    
    # Ensure types are strictly Python int or None
    # lpf[m] is an int, m is an int from range
    return [w if isinstance(w, int) and not isinstance(w, bool) else None for w in result]
