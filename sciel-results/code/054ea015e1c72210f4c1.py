import math

def _phi(n):
    """Compute Euler's totient function for n."""
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a direct search with efficient totient calculation.
    """
    # The mathematical minimum is known to be 143.
    # However, we compute it to be robust and satisfy the "stronger algorithm" directive.
    # We can optimize by checking only numbers that have prime factors p where p-1 is divisible by 5 (for the factor 5 in 120).
    # Primes p such that p % 5 == 1: 11, 31, 41, 61, ...
    # Or n is a multiple of 25.
    
    # Given the small range, a direct loop is fast enough (< 1ms).
    for n in range(1, 1000001):
        if _phi(n) % 120 == 0:
            return n
            
    return -1 # Should not reach here
