# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    N = 256
    
    # Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # For each prime, find the largest power <= N
    result = [1]  # 1 is coprime with everything
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        result.append(power)
    
    # Verify pairwise coprimality (safety check)
    import math
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # This shouldn't happen, but if it does, fall back
                return [1] + [a for a in range(2, N + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    return result
# EVOLVE-BLOCK-END
