"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    # We need to find the minimum n such that 120 | phi(n).
    # phi(n) = product over primes p|n of p^(a-1)(p-1).
    # For 120 = 2^3 * 3 * 5 to divide phi(n), the factors (p-1) and p^(a-1) must cover these.
    
    # Let's do a direct search for the smallest n.
    # We can compute phi(n) for n from 1 upwards until we find one divisible by 120.
    # To be efficient, we can use a sieve to compute phi(n) for all n up to a limit.
    # However, since we want the *smallest* n, we can just iterate n=1,2,3... and compute phi(n) on the fly?
    # No, computing phi(n) for each n individually is O(sqrt(n)) or O(number of factors).
    # A sieve up to, say, 10000 is very fast.
    # Let's check how small the answer is.
    # phi(143) = phi(11)*phi(13) = 10*12 = 120. So 143 is a candidate.
    # Is there a smaller one?
    # Let's just sieve up to a reasonable limit, e.g., 10000.
    # If we don't find it, we can increase the limit, but 143 is likely the answer or close.
    
    limit = 10000
    phi = list(range(limit + 1))
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    # If not found (unlikely), return a known valid large number or extend search
    # But the problem says n in [1, 1000000], and 143 is definitely < 10000.
    return 143
# EVOLVE-BLOCK-END
