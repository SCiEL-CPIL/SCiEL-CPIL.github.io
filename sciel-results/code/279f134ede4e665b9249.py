P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    import time
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Precompute primes up to max gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # Sieve for primes up to max_gap
    sieve = [True] * (max_gap + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_gap**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_gap + 1, i):
                sieve[j] = False
    primes_below = [p for p in range(2, max_gap) if sieve[p]]
    
    # For each gap, find a witness
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Primes strictly less than gap that we need to check
        relevant_primes = [p for p in primes_below if p < gap]
        
        found = False
        for m in range(lo + 1, hi):
            # Check if m has any prime factor < gap
            is_rough = True
            for p in relevant_primes:
                if m % p == 0:
                    is_rough = False
                    break
            if is_rough:
                result[i] = m
                found = True
                break
        
        if not found:
            result[i] = None
    
    return result
