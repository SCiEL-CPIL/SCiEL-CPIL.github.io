import math
from functools import reduce
from itertools import product

def phi(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n = n // p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

def construct():
    # Direct constructive search using divisors of 120
    # We want to find minimal n such that phi(n) is divisible by 120
    # phi(n) = product of phi(p_i^k_i) where n = product of p_i^k_i
    # For prime p, phi(p^k) = p^(k-1)*(p-1)
    # We need phi(n) % 120 == 0
    
    # First, let's find the minimal n by direct search up to a reasonable limit
    # Since we know the answer is small, we can just check from 1 upward
    
    # Actually, let's do a smarter approach:
    # We know phi(n) must be divisible by 120 = 2^3 * 3 * 5
    # Let's search n from 1 to 1000000 and find the first one where phi(n) % 120 == 0
    
    # For efficiency, we'll just iterate and compute phi
    for n in range(1, 1000001):
        if phi(n) % 120 == 0:
            return n
    
    # This should never be reached
    return 1
