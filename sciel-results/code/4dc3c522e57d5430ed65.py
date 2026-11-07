"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Find the maximum value needed for the sieve
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Sieve for Least Prime Factor (LPF)
    # lpf[i] will store the smallest prime factor of i
    # If lpf[i] == 0, i is prime (or 1, handled separately)
    lpf = [0] * (max_val + 1)
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, max_val + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    result = []
    for lo, hi in P['gaps']:
        gap_size = hi - lo
        witness = None
        for m in range(lo + 1, hi):
            # Check if the least prime factor of m is >= gap_size
            # Note: if m is prime, lpf[m] = m. If m < gap_size, this fails.
            # If m is composite, lpf[m] is its smallest prime factor.
            # Condition: least prime factor >= gap_size
            if m == 1:
                # 1 has no prime factors, so technically "least prime factor" is undefined/infinite?
                # But 1 is never in the range [10008, ...] so this is safe.
                continue
            if lpf[m] >= gap_size:
                witness = m
                break
        result.append(witness)
    
    return result
# EVOLVE-BLOCK-END
