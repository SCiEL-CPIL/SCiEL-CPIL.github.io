"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Determine the maximum value needed for the sieve
    max_val = max(hi for _, hi in P['gaps'])
    
    # Sieve for Least Prime Factor (LPF)
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i] = i
            # Mark multiples of i
            if i * i <= max_val:
                for j in range(i * i, max_val + 1, i):
                    if lpf[j] == 0:
                        lpf[j] = i
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap_size = hi - lo
        # Use boolean array to mark invalid numbers (those with lpf < gap_size)
        # A number is valid if its least prime factor >= gap_size
        # i.e., it is not divisible by any prime < gap_size
        
        # Find all primes less than gap_size
        primes_below = [p for p in range(2, gap_size) if lpf[p] == p]
        
        # Create a boolean array for the range (lo, hi)
        # valid[m - lo - 1] indicates whether m is a valid witness
        length = hi - lo - 1
        if length <= 0:
            continue
        valid = [True] * length
        
        # Mark all multiples of each prime < gap_size as invalid
        for p in primes_below:
            # Find the first multiple of p in (lo, hi)
            start = ((lo + 1 + p - 1) // p) * p
            for m in range(start, hi, p):
                valid[m - lo - 1] = False
        
        # Find the first valid number
        for idx, is_valid in enumerate(valid):
            if is_valid:
                result[i] = lo + 1 + idx
                break
    
    return result
# EVOLVE-BLOCK-END
