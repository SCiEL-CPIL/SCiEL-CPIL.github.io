import math

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) % 120 == 0.
    Uses a Smallest Prime Factor (SPF) sieve to compute phi(n) in O(N).
    """
    limit = 1000000
    
    # Smallest Prime Factor (SPF) sieve
    # spf[i] will store the smallest prime factor of i
    spf = list(range(limit + 1))
    
    # Standard SPF sieve
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute phi(n) for all n up to limit
    phi = [0] * (limit + 1)
    phi[1] = 1
    
    for n in range(2, limit + 1):
        p = spf[n]
        if p == n:
            phi[n] = n - 1
        else:
            m = n // p
            if m % p == 0:
                phi[n] = phi[m]
            else:
                phi[n] = phi[m] * (p - 1)
        
        if phi[n] % 120 == 0:
            return n

    return limit
