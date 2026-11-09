"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The optimal strategy for maximizing the number of adjacent coprime pairs
    # in the sorted divisor list is to use the product of the first k distinct primes
    # such that the product is <= 10^18.
    # For square-free n with k primes, the score is 2^(k-1).
    # We need the largest k such that primorial(k) <= 10^18.
    
    # First 16 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    n = 1
    for p in primes:
        if n * p <= 10**18:
            n *= p
        else:
            break
            
    return n
# EVOLVE-BLOCK-END
