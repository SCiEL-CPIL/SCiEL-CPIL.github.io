import time

def _phi(n):
    """Calculate Euler's totient function for n."""
    result = n
    p = 2
    x = n
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p += 1
    if x > 1:
        result -= result // x
    return result

def construct():
    # The minimal n is expected to be small. 
    # Searching up to 1000 is sufficient and very fast.
    for n in range(1, 1000):
        if _phi(n) % 120 == 0:
            return n
    # Fallback if not found (should not happen)
    return 14400
