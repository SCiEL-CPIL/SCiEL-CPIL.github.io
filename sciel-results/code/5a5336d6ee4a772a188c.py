"""
Rewritten program using direct divisibility checks for witness validation.
This avoids precomputation and potential sieve bugs, relying on the small size of the gaps.
"""

def construct():
    """
    Finds integer witnesses m in (lo, hi) such that least_prime_factor(m) >= gap.
    Uses direct divisibility checks by primes less than the gap.
    """
    # Primes up to 30 are sufficient for gaps up to 28 (primes < 28 are 2,3,5,7,11,13,17,19,23)
    # We need to check divisibility by primes p such that p < gap.
    # Max gap is 28. Primes < 28: 2, 3, 5, 7, 11, 13, 17, 19, 23.
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        witness = None
        
        # Check all integers strictly between lo and hi
        for m in range(lo + 1, hi):
            # Check if m has any prime factor < gap
            # If it does, LPF(m) < gap, so it's not a witness.
            # If it doesn't, LPF(m) >= gap (either m is prime, or its smallest factor is >= gap).
            valid = True
            for p in small_primes:
                if p >= gap:
                    break
                if m % p == 0:
                    valid = False
                    break
            
            if valid:
                witness = m
                break
        
        if witness is not None:
            result[i] = int(witness)
            
    return result
