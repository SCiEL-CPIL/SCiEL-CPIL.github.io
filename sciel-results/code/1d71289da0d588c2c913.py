import math
from typing import List

def _totient(n: int) -> int:
    """Compute Euler's totient phi(n) exactly via factorization."""
    if n <= 0:
        raise ValueError("n must be positive")
    original = n
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

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    This is a constant-time, deterministic solution: we know analytically that
    phi(143) = phi(11 * 13) = 10 * 12 = 120, so 143 is a valid candidate.
    
    To be rigorous, we verify that no smaller n in [2, 142] has phi(n) divisible by 120.
    This brute-force check is O(142 * sqrt(142)) which is negligibly fast.
    
    If for any reason the verification finds a smaller n (which it won't), we return that.
    Otherwise, we return 143.
    """
    # Verify that 143 works: phi(143) = 120, which is divisible by 120
    if _totient(143) % 120 != 0:
        # This should never happen, but as a safety net
        # Expand search if our analytical answer is wrong
        for n in range(144, 1000001):
            if _totient(n) % 120 == 0:
                return n
        return 143  # Should never reach here
    
    # Verify minimality: check all n in [2, 142] to ensure none has phi(n) divisible by 120
    for n in range(2, 143):
        if _totient(n) % 120 == 0:
            return n
    
    return 143
