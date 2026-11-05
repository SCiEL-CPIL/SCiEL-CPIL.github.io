"""Rewritten program to find the smallest n with phi(n) divisible by 120."""
# EVOLVE-BLOCK-START
import math
import time

def _phi(n):
    """
    Compute Euler's totient function phi(n) using trial division.
    Efficient for small n (n < 10000).
    """
    if n == 1:
        return 1
    result = n
    p = 2
    # Check for factor 2
    if n % p == 0:
        result -= result // p
        while n % p == 0:
            n //= p
    
    # Check for odd factors
    p = 3
    while p * p <= n:
        if n % p == 0:
            result -= result // p
            while n % p == 0:
                n //= p
        p += 2
    
    # If n is still greater than 1, then it is a prime factor
    if n > 1:
        result -= result // n
        
    return result

def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Iterates n from 1 upwards and checks the condition.
    """
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # We expect the answer to be small, so we iterate upwards.
    # Upper bound is 1,000,000 as per task contract, but we will likely stop much earlier.
    for n in range(1, 1000001):
        # Check time to be safe, though this loop is extremely fast for small n
        if time.time() > deadline:
            break
            
        phi_n = _phi(n)
        if phi_n % 120 == 0:
            return n
            
    # Fallback (shouldn't reach here as a solution exists well within 1000)
    return 1000000
# EVOLVE-BLOCK-END
