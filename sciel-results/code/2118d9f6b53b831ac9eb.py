import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}
    
    # Precompute small primes up to max gap
    # Max gap in list is 28. Let's compute primes up to 30 to be safe.
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = get_primes(max_gap)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # We need primes q such that q < gap
        # If gap <= 2, there are no primes < 2, so any m works.
        divisors = [p for p in small_primes if p < gap]
        
        found = False
        # Iterate through candidates in the gap
        for m in range(lo + 1, hi):
            # Check if m is divisible by any prime < gap
            is_valid = True
            for q in divisors:
                if m % q == 0:
                    is_valid = False
                    break
            
            if is_valid:
                result[i] = m
                found = True
                break
        
        # If not found, result[i] remains None
        # The problem states "None never asserts nonexistence", but we want to maximize non-None.
        # Given the density, we should find one for all.
        
    return result
