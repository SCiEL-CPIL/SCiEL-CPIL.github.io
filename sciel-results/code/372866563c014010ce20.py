"""Construct an integer n <= 10^18 maximizing the number of adjacent coprime divisor pairs."""
# EVOLVE-BLOCK-START
def construct():
    # The optimal strategy is to use the product of the smallest primes (primorial)
    # to maximize the number of distinct prime factors within the 10^18 limit.
    # This maximizes the "structural diversity" of the divisors, leading to
    # a high number of adjacent pairs that are coprime in the sorted divisor list.
    
    limit = 10**18
    n = 1
    
    # First 20 primes are more than enough since 2*3*5*...*71 > 10^18
    # We only need to check the first ~15 primes.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    for p in primes:
        if n * p <= limit:
            n *= p
        else:
            break
            
    return n
# EVOLVE-BLOCK-END
