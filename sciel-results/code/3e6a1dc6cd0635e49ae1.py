import math

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a targeted search for small n first, falling back to a sieve if necessary.
    """
    target = 120
    
    # Fast path: Check small n up to 1000
    # This covers the expected minimum (143) very quickly.
    for n in range(1, 1001):
        if _totient(n) % target == 0:
            return n
            
    # Fallback: Sieve for larger n if not found in small range
    # This ensures correctness if the minimum is > 1000
    N = 1000000
    phi = list(range(N + 1))
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    for n in range(1001, N + 1):
        if phi[n] % target == 0:
            return n
            
    # Fallback if no solution found (should not happen)
    return 1

def _totient(n):
    """
    Compute Euler's totient function for a single integer n.
    """
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result
