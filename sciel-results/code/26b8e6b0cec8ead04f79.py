"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 256
    # Generate primes up to n using Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    result = [1]
    for p in primes:
        val = p
        # Calculate the largest power of p that is <= n
        while val * p <= n:
            val *= p
        result.append(val)
        
    return result

# EVOLVE-BLOCK-END
