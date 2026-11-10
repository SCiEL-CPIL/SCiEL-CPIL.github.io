import time

def _phi(n):
    """Calculate Euler's totient function for n."""
    if n == 1:
        return 1
    result = n
    x = n
    # Handle small primes separately for speed
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if p * p > x:
            break
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
    if x > 1:
        result -= result // x
    return result

def construct():
    # The minimal n with phi(n) divisible by 120 is small.
    # Search from 1 upward; the answer is found quickly.
    for n in range(1, 1000):
        if _phi(n) % 120 == 0:
            return n
    # Fallback (should not happen)
    return 14400
