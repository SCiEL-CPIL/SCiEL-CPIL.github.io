"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Determine the range we need to cover
    lo_bound = min(lo for lo, hi in P['gaps'])
    hi_bound = max(hi for lo, hi in P['gaps'])
    
    # We need LPF for integers in [lo_bound+1, hi_bound-1]
    # Actually, let's compute LPF for all integers up to hi_bound
    n = hi_bound
    
    # Sieve to compute least prime factor for each number up to n
    lpf = list(range(n + 1))
    lpf[0] = 0
    lpf[1] = 1
    for i in range(2, int(n**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i*i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        witness = None
        
        # Scan candidates in (lo, hi) for one with lpf >= gap
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                witness = m
                break
        
        result[i] = witness
    
    return result

# EVOLVE-BLOCK-END
