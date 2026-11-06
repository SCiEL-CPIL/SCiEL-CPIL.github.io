"""Rewritten program: Maximize coprime adjacent divisor pairs using Primorial."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns an integer n <= 10^18 that maximizes the number of adjacent 
    coprime pairs in its sorted list of divisors.
    
    Based on analysis, square-free numbers (product of distinct primes) 
    maximize the "mixing" of prime factors in the divisor sequence, 
    leading to more gcd(d_i, d_{i+1}) == 1. 
    Among square-free numbers, using the smallest primes (Primorial) 
    appears to yield the highest score for a given number of prime factors.
    The maximum number of distinct prime factors for n <= 10^18 is 15.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # The first 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    n = 1
    for p in primes:
        n *= p
        
    # n is P_15 = 614889782588491410
    # This is < 10^18
    if n <= 10**18:
        return n
    
    # Fallback if calculation goes wrong (should not happen)
    return 2
# EVOLVE-BLOCK-END
