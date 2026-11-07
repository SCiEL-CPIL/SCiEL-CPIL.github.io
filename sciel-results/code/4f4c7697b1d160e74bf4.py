"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    from math import isqrt
    
    # Find all primes in the range [2, n]
    primes = []
    for a in range(2, P['n'] + 1):
        is_prime = True
        for d in range(2, isqrt(a) + 1):
            if a % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(a)
            
    # The maximum number of linearly independent vectors in the prime exponent space
    # is the number of primes <= n.
    # We return these primes, which guarantees distinct subset products.
    return primes
# EVOLVE-BLOCK-END
