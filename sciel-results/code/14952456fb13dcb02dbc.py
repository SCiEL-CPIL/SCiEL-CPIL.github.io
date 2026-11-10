# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # 1. Sieve for primes up to MAX_N
    # We need primes up to MAX_N to identify p and p^2
    limit = MAX_N + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. Compute S(n) = product of p where p|n and p^2 !| n
    S = np.ones(limit + 1, dtype=np.int64)
    
    for p in primes:
        if p > MAX_N:
            break
        p2 = p * p
        # Multiples of p
        multiples = np.arange(p, limit + 1, p)
        # We need to exclude multiples of p^2
        # Multiples of p^2 within range
        if p2 <= MAX_N:
            multiples_p2 = np.arange(p2, limit + 1, p2)
            # Create a boolean mask for 'not multiple of p^2'
            # Easier: set S[m] *= p for all multiples of p, then we need to "divide" or handle logic differently?
            # No, S(m) is product of p where v_p(m)=1.
            # So for each p, we want to multiply S[m] by p if m is multiple of p but NOT p^2.
            
            # Using numpy sets might be slow for large arrays if done naively.
            # Let's use a boolean array for "is_multiple_of_p2"
            # Or simply:
            # S[multiples] *= p
            # S[multiples_p2] //= p  <-- This works because p is prime and we only multiplied by p once?
            # Wait. If m is multiple of p^2, it is also multiple of p.
            # So S[m] was multiplied by p. We want it to NOT be multiplied by p.
            # So dividing by p is correct.
            
            S[multiples] *= p
            S[multiples_p2] //= p
        else:
            # p^2 > MAX_N, so no multiple of p in [1, MAX_N] is a multiple of p^2
            S[multiples] *= p

    # 3. Compute PP(n) = n // S(n)
    # Ensure n is not 0
    n_range = np.arange(1, limit + 1)
    PP = n_range // S
    
    # 4. Find n in [MIN_N, MAX_N] maximizing PP[n] * PP[n+1]
    # We need PP[n] * PP[n+1] for n in [MIN_N, MAX_N]
    # Note: PP array is 1-indexed in our logic (indices 0..limit)
    # PP[0] is 0 // 1 = 0.
    
    # Calculate products
    # n goes from MIN_N to MAX_N
    # We need PP[n] and PP[n+1]
    # Slice PP[MIN_N : MAX_N+1] and PP[MIN_N+1 : MAX_N+2]
    
    prod = PP[MIN_N:MAX_N+1] * PP[MIN_N+1:MAX_N+2]
    
    # Find index of max
    max_idx = np.argmax(prod)
    best_n = MIN_N + max_idx
    
    # Verify time
    if time.time() > deadline:
        # Fallback to a simple check or just return best found so far?
        # The numpy ops should be very fast (<1s)
        pass
        
    return int(best_n)
# EVOLVE-BLOCK-END
