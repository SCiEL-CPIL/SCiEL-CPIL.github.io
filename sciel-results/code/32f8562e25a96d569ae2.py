import math
import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # seconds
    
    # Sieve of Eratosthenes to find all primes up to TARGET
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(TARGET**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Precompute largest prime factor for all numbers up to TARGET
    # lp[n] = largest prime factor of n
    lp = [0] * (TARGET + 1)
    lp[1] = 1  # P(1) = 1
    
    for p in primes:
        # For each prime p, mark all multiples of p
        # Since we iterate primes in increasing order, the last prime that divides n
        # will be the largest prime factor
        for multiple in range(p, TARGET + 1, p):
            lp[multiple] = p
    
    # Now lp[n] contains the largest prime factor of n for all n in [1, TARGET]
    
    # Find the minimum value of max(lp[a], lp[b]) where a + b = TARGET
    # and a, b >= 1
    
    best_max_pf = float('inf')
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET):
        b = TARGET - a
        max_pf = max(lp[a], lp[b])
        if max_pf < best_max_pf:
            best_max_pf = max_pf
            best_a = a
            best_b = b
            # Early termination: if we found max_pf = 2, that's the minimum possible
            # (since we need positive integers, and 2 is the smallest prime)
            # Actually, we can't get below 2 unless one of them is 1
            # If a=1, lp[1]=1, so max(1, lp[TARGET-1]) = lp[TARGET-1]
            # If TARGET-1 is a power of 2, then lp[TARGET-1] = 2
            if best_max_pf <= 2:
                break
    
    # Check if we hit the deadline (shouldn't for this small problem, but just in case)
    if time.time() - start_time > DEADLINE:
        pass  # We already have our answer
    
    return [best_a, best_b]
