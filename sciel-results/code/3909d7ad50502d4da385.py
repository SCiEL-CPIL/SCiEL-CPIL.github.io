"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def sieve(n):
    """Generate a list of primes up to n."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Precompute primes up to the maximum possible gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # We need primes strictly less than the gap g to check if they divide m.
    # If m is not divisible by any prime < g, then lpf(m) >= g.
    # Note: if m is prime, lpf(m) = m. But m is between consecutive primes, so m is not prime.
    # So lpf(m) is a prime factor of m.
    # We need lpf(m) >= g.
    # This is equivalent to: m is not divisible by any prime p < g.
    
    primes_up_to_max = sieve(max_gap)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # Get relevant primes: all primes p such that p < gap
        relevant_primes = [p for p in primes_up_to_max if p < gap]
        
        found = False
        # Search for m in (lo, hi)
        for m in range(lo + 1, hi):
            is_rough = True
            for p in relevant_primes:
                if m % p == 0:
                    is_rough = False
                    break
            if is_rough:
                # m is not divisible by any prime < gap
                # Since m is not prime (between consecutive primes), its least prime factor must be >= gap
                result[i] = m
                found = True
                break
                
        if not found:
            # Optional: deeper search if simple check fails, but likely unnecessary for these small gaps
            # The simple check is exhaustive for the condition "lpf(m) >= gap"
            pass
            
    return result

# EVOLVE-BLOCK-END
