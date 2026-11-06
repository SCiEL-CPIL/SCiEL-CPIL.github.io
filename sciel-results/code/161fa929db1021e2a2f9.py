import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    min_n = 10000
    max_n = 1000000
    limit = max_n + 1
    
    # Compute primes up to sqrt(limit) for trial division of large numbers
    sqrt_limit = int(math.isqrt(limit))
    
    # Sieve of Eratosthenes for primes up to sqrt_limit
    sieve = np.ones(sqrt_limit + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    small_primes = np.nonzero(sieve)[0]
    
    # Initialize lpf array
    lpf = np.zeros(limit + 1, dtype=np.int32)
    
    # For numbers in [2, limit], find largest prime factor
    # Strategy: 
    # 1. For small primes, mark multiples
    # 2. For numbers not fully factored by small primes, they have a large prime factor
    #    which is either the number itself (if prime) or a large prime
    
    # Mark multiples of small primes
    for p in small_primes:
        for multiple in range(p, limit + 1, p):
            lpf[multiple] = p
    
    # Now, for numbers where lpf is still small, we need to check if they have a larger prime factor
    # Actually, the above approach marks the SMALLEST prime factor initially if we iterate in order
    # Wait, no: if we iterate p in increasing order and set lpf[multiple] = p, then the last prime that divides it wins.
    # But primes are iterated in increasing order, so the LARGEST prime factor will be the last one to set it.
    # So this correctly computes the largest prime factor for all numbers that are divisible by at least one small prime.
    # For numbers NOT divisible by any small prime, lpf is still 0. These numbers must be either:
    #   - 1 (not in our range)
    #   - prime (and > sqrt_limit)
    #   - product of two primes both > sqrt_limit (impossible since (sqrt_limit)^2 = limit, so product > limit)
    #   - Actually, a number n <= limit can have at most one prime factor > sqrt_limit.
    #   - If n is not divisible by any prime <= sqrt_limit, then n is prime (or 1).
    #   - Wait, what if n = p * q where p and q are both > sqrt_limit? Then p*q > limit, impossible.
    #   - So if n is not divisible by any small prime, n is prime.
    #   - Therefore, lpf[n] should be n for these numbers.
    
    # But wait: the loop above sets lpf[multiple] = p for each prime p. Since p is increasing, the last p that divides multiple is the largest.
    # So for numbers divisible by at least one small prime, lpf is correct.
    # For numbers not divisible by any small prime, lpf is 0, and they are prime, so lpf[n] = n.
    
    # Find indices where lpf is 0 (excluding 0 and 1)
    zero_indices = np.nonzero(lpf == 0)[0]
    zero_indices = zero_indices[zero_indices >= 2]
    for idx in zero_indices:
        lpf[idx] = idx
    
    if time.time() > deadline:
        return min_n
    
    # Compute M(n) = max(lpf[n], lpf[n+1]) for n in [min_n, max_n]
    left = lpf[min_n:max_n + 1]
    right = lpf[min_n + 1:max_n + 2]
    
    max_factors = np.maximum(left, right)
    min_idx = np.argmin(max_factors)
    
    best_n = min_n + min_idx
    
    return int(best_n)
