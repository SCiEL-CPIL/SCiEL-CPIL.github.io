"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The score is maximized by using the maximum number of distinct primes (squarefree n)
    # because each new prime roughly doubles the number of divisors and adds many new
    # coprime adjacent pairs, whereas increasing exponents adds fewer such pairs.
    # We use the smallest primes to stay within the 10^18 limit.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    n = 1
    for p in primes:
        n *= p
        if n > 10**18:
            # If we exceeded the limit, we would need to backtrack, 
            # but for the first 15 primes, we are safe.
            # This guard is for robustness if the list is extended.
            n //= p
            break
            
    return n
# EVOLVE-BLOCK-END
