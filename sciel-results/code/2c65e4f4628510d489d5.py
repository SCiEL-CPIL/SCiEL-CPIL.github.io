"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    import time
    deadline = time.time() + 75  # 75s safety margin out of 90s limit
    
    # Precompute primes up to a reasonable limit for least-factor checking
    # The maximum gap is 28 (10273 to 10289), so we need primes up to ~28
    # But for safety, let's go up to maybe 1000 to handle any edge cases
    # Actually, the least prime factor of m must be >= gap. 
    # The gap is at most 28. So we need to check divisibility by primes up to sqrt(m) ~ 100.
    # But the condition is: least prime factor >= gap.
    # This means m has no prime factor < gap.
    # So we just need to check if m is divisible by any prime < gap.
    # If not, then its least prime factor is >= gap (or m is prime, in which case least factor is m >= gap since m > lo and gap is small).
    
    # Let's precompute all primes up to 1000 (more than enough for sqrt(10289) ~ 101)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(200)  # primes up to 200
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # We need an integer m in (lo, hi) such that least prime factor of m >= gap.
        # This is equivalent to: m is not divisible by any prime < gap.
        # Primes less than gap: filter from our prime list
        small_primes = [p for p in primes if p < gap]
        
        # Search for m in range(lo+1, hi)
        # For efficiency, try from both ends or just sequentially
        # Let's try sequentially from lo+1 to hi-1
        for m in range(lo + 1, hi):
            # Check if m has any prime factor < gap
            has_small_factor = False
            for p in small_primes:
                if m % p == 0:
                    has_small_factor = True
                    break
            if not has_small_factor:
                result[i] = m
                break
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Ensure types are correct
    final_result = []
    for item in result:
        if item is None:
            final_result.append(None)
        else:
            final_result.append(int(item))
    
    return final_result
# EVOLVE-BLOCK-END
