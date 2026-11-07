"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    result = [None] * len(P['gaps'])
    
    # Precompute primes up to sqrt(max_hi) for efficient LPF checking
    # Max hi is 10289, sqrt is ~101.
    max_val = max(hi for _, hi in P['gaps'])
    limit = int(math.isqrt(max_val)) + 1
    
    # Sieve of Eratosthenes for primes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # We need m in (lo, hi) such that least prime factor of m >= gap
        found = False
        for m in range(lo + 1, hi):
            # Check least prime factor
            # If m is prime, LPF is m. If m < gap, this fails (but m > lo >= 10000, gap <= 28, so m > gap always).
            # So if m is prime, LPF(m) = m >= gap is always true.
            # If m is composite, we find its smallest prime factor.
            
            lpf = m
            for p in primes:
                if p * p > m:
                    break
                if m % p == 0:
                    lpf = p
                    break
            
            if lpf >= gap:
                result[i] = m
                found = True
                break
                
        # If not found, result[i] remains None
        # No early return here! We must check all gaps.
        
    return result
# EVOLVE-BLOCK-END
