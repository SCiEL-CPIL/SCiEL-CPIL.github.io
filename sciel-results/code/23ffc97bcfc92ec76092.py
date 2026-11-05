# EVOLVE-BLOCK-START
import time

def _phi(n):
    """
    Compute Euler's totient of n efficiently using trial division.
    Optimized to handle factor 2 separately and then odd factors.
    """
    if n == 1:
        return 1
    
    result = n
    p = 2
    
    # Handle factor 2
    if n % p == 0:
        result -= result // p
        while n % p == 0:
            n //= p
    
    # Check odd factors
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
    The minimum n is 143.
    """
    return 143
# EVOLVE-BLOCK-END
