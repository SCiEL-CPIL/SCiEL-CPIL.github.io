import time
import math

def construct():
    """
    Returns a list of distinct ints from 1..256, pairwise coprime, maximizing sum.
    The optimal solution is {1} U {p^k | p is prime, p^k <= 256 < p^(k+1)}.
    """
    N = 256
    
    # Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    result = [1]
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        result.append(power)
        
    return result
