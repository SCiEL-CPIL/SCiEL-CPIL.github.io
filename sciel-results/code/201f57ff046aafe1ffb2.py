"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute small primes up to max gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    primes = []
    for i in range(2, max_gap):
        is_p = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(i)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Filter primes less than gap
        small_primes = [p for p in primes if p < gap]
        
        found = False
        # Check all integers strictly between lo and hi
        for m in range(lo + 1, hi):
            # Check if m is divisible by any small prime
            valid = True
            for p in small_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = int(m)
                found = True
                break
                
    return result
# EVOLVE-BLOCK-END
