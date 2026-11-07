"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    n = 256
    # Sieve of Eratosthenes to find primes up to 256
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    result = [1]
    
    for p in range(2, n + 1):
        if is_prime[p]:
            # Find the highest power of p that is <= n
            val = p
            while val * p <= n:
                val *= p
            result.append(val)
            
    return result
# EVOLVE-BLOCK-END
