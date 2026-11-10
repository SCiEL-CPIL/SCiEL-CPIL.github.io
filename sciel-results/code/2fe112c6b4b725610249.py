"""Smooth two-summand decomposition: minimize max(P(a), P(b)) for a+b=123457."""
import time
import math

TARGET = 123457

def construct():
    """
    Return [a, b] with a + b = 123457 minimizing max(P(a), P(b)),
    where P(n) is the largest prime factor of n (P(1)=1).
    
    Strategy: Precompute largest prime factor for all numbers up to TARGET
    using a modified sieve, then scan all splits to find the global optimum.
    """
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Step 1: Sieve to compute largest prime factor for each number up to TARGET
    # largest_prime_factor[i] = largest prime factor of i
    lp = [0] * (TARGET + 1)
    
    # Standard sieve approach: for each prime p, mark all multiples with p
    # We want the LARGEST prime factor, so we iterate primes in increasing order
    # and overwrite — the last prime that divides n is the largest.
    
    # Actually, a cleaner approach: 
    # For each prime p, for each multiple m of p, set lp[m] = p.
    # Since we process primes in increasing order, the final value is the largest.
    
    # First, generate primes up to TARGET using a simple sieve
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(TARGET)) + 1):
        if is_prime[i]:
            for j in range(i * i, TARGET + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, TARGET + 1) if is_prime[i]]
    
    # Now for each prime, update lp for its multiples
    for p in primes:
        for m in range(p, TARGET + 1, p):
            lp[m] = p
    
    # lp[1] = 1 by definition
    lp[1] = 1
    
    # Step 2: Scan all splits a + b = TARGET, a, b >= 1
    best_max = float('inf')
    best_pair = (1, TARGET - 1)
    
    for a in range(1, TARGET):
        b = TARGET - a
        pa = lp[a]
        pb = lp[b]
        mx = pa if pa > pb else pb
        if mx < best_max:
            best_max = mx
            best_pair = (a, b)
            # Early termination: if best_max == 2, we can't do better than 2
            # (since both must be positive integers > 1 for P > 1, 
            #  the minimum possible max P is 2, achieved only if both are powers of 2)
            if best_max <= 2:
                break
    
    # Check if we hit the deadline (shouldn't for this size, but safety)
    if time.time() > deadline:
        # Return best found so far
        return [best_pair[0], best_pair[1]]
    
    return [best_pair[0], best_pair[1]]
