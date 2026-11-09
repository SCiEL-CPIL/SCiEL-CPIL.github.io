"""Improved program: Precompute LPF using a modified Sieve of Eratosthenes approach for clarity and distinct design, then scan gaps for witnesses."""

# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _compute_lpf_sieve(limit):
    """
    Computes the Least Prime Factor (LPF) for all numbers up to limit.
    Uses a sieve approach where we mark the smallest prime factor for composites.
    """
    lpf = [0] * (limit + 1)
    primes = []
    
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i] = i
            primes.append(i)
        
        for p in primes:
            if p > lpf[i] or p * i > limit:
                break
            lpf[p * i] = p
            
    return lpf


def construct():
    # Find the upper bound for the sieve
    max_val = max(hi for _, hi in P['gaps'])
    
    # Precompute LPF table
    lpf_table = _compute_lpf_sieve(max_val)
    
    witnesses = []
    for lo, hi in P['gaps']:
        gap_size = hi - lo
        witness = None
        
        # Search for a number m in (lo, hi) such that lpf[m] >= gap_size
        for m in range(lo + 1, hi):
            if lpf_table[m] >= gap_size:
                witness = m
                break
        
        witnesses.append(witness)
        
    # Ensure types are strictly Python int or None
    return [w if w is not None else None for w in witnesses]
# EVOLVE-BLOCK-END
