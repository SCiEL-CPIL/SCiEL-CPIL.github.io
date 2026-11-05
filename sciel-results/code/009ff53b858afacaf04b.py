import math

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs.
    
    The optimal strategy for n = 255 is to take all multiples of the 
    smallest prime factor of 255, which is 3.
    Size: 85.
    """
    n = 255
    
    # Find prime factors of n
    def prime_factors(num):
        factors = []
        d = 2
        while d * d <= num:
            if num % d:
                d += 1
            else:
                factors.append(d)
                while num % d == 0:
                    num //= d
        if num > 1:
            factors.append(num)
        return factors

    # Get smallest prime factor
    pf = prime_factors(n)
    p_min = min(pf)
    
    # Candidate set: multiples of p_min
    # This set contains n because p_min divides n.
    # All elements share p_min, so gcd >= p_min > 1.
    candidate = list(range(p_min, n + 1, p_min))
    
    return candidate
