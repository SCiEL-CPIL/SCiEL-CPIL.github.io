"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _compute_lpf_array(n):
    """Compute least prime factor for all integers from 0 to n using a segmented-style sieve.
    Returns a list lpf where lpf[m] is the least prime factor of m, or m itself if m is prime.
    lpf[0] and lpf[1] are set to 0 (unused)."""
    lpf = list(range(n + 1))
    if n >= 0:
        lpf[0] = 0
    if n >= 1:
        lpf[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    return lpf

# Precompute LPF array covering the full range of all pairs
_MAX_VAL = max(hi for lo, hi in P['gaps'])
_LPF = _compute_lpf_array(_MAX_VAL)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds margin
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        witness = None
        # Scan precomputed LPF array for m in (lo, hi) where lpf[m] >= gap
        for m in range(lo + 1, hi):
            if _LPF[m] >= gap:
                witness = m
                break
        
        result[i] = witness
    
    return result

# EVOLVE-BLOCK-END
