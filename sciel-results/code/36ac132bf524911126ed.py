"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The optimal strategy is to maximize the number of distinct prime factors
    # to maximize the "interleaving" of divisors, which increases the count
    # of adjacent coprime pairs. We use the first k primes such that their
    # product is <= 10^18.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    n = 1
    for p in primes:
        if n * p <= 10**18:
            n *= p
        else:
            break
            
    return n
# EVOLVE-BLOCK-END
