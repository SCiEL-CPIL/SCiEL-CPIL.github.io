"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The set of integers is derived from prime power chains to maximize count
    # while ensuring distinct subset products.
    # 
    # For prime 2, powers <= 40 are 2, 4, 8, 16, 32.
    # Exponents 1, 2, 3, 4, 5.
    # Max distinct subset sum set: {1, 2, 4} -> numbers 2, 4, 16.
    #
    # For prime 3, powers <= 40 are 3, 9, 27.
    # Exponents 1, 2, 3.
    # Max distinct subset sum set: {1, 3} -> numbers 3, 27. (1,2 fails as 1+2=3)
    #
    # For prime 5, powers <= 40 are 5, 25.
    # Exponents 1, 2.
    # Max distinct subset sum set: {1, 2} -> numbers 5, 25.
    #
    # For primes 7, 11, 13, 17, 19, 23, 29, 31, 37, only the prime itself fits.
    #
    # Total count: 3 + 2 + 2 + 9 = 16.
    
    result = [
        2, 4, 16,  # 2^1, 2^2, 2^4
        3, 27,     # 3^1, 3^3
        5, 25,     # 5^1, 5^2
        7, 11, 13, 17, 19, 23, 29, 31, 37
    ]
    
    return result
# EVOLVE-BLOCK-END
