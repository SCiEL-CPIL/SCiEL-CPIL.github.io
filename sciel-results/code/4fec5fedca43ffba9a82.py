import math
import time
import random

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 130s usage
    DEADLINE = start_time + 130.0
    
    # 1. Precompute primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    if limit_p >= 1:
        is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Optimized phi calculation with early termination
    # Precompute small primes for faster trial division
    small_primes = primes[:1000]  # primes up to ~7900
    
    def phi(x):
        if x == 1: return 1
        n = x
        result = x
        # Trial division by small primes first
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        # If remaining n is large, check if it's prime
        if n > 1:
            # For large n, we need to check if it's prime
            # If n is prime, we're done
            # If not, we need to factor it further
            # Use a quick primality test for the remaining factor
            if n < limit_p:
                if is_prime[n]:
                    result -= result // n
            else:
                # n is large, assume it might be prime or have large factors
                # Do a more careful check
                limit = int(n ** 0.5)
                for p in primes:
                    if p > limit:
                        break
                    if n % p == 0:
                        while n % p == 0:
                            n //= p
                        result -= result // p
                        limit = int(n ** 0.5)
                if n > 1:
                    result -= result // n
        return result

    best_k = 1
    best_s = 1
    
    # 4. Search Strategy with improved speed
    # We want to find a start s such that phi(s), ..., phi(s+k-1) are distinct.
    # Random sampling of s in [1, 10^12].
    # For each s, we compute phi(s), phi(s+1), ... until a collision or limit k=200 or time out.
    
    # Helper to check a specific start with time limit
    # Use a more efficient approach: batch process and use a dict for seen values
    def check_start(s, deadline):
        seen = {}
        k = 0
        limit_k = 200
        while k < limit_k:
            if time.time() > deadline:
                break
            val = phi(s + k)
            if val in seen:
                break
            seen[val] = True
            k += 1
        return k
    
    # Phase 1: Check small ranges for quick baseline
    # Also check some strategically chosen larger starts
    small_starts = [1, 2, 3, 4, 5, 10, 100, 1000, 10000]
    for s in small_starts:
        if time.time() > DEADLINE - 10:
            break
        k = check_start(s, DEADLINE)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 1b: Try some medium-sized starts
    medium_starts = [10**6, 10**7, 10**8, 10**9, 10**10, 10**11]
    for s in medium_starts:
        if time.time() > DEADLINE - 10:
            break
        k = check_start(s, DEADLINE)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Random Search with improved efficiency
    # Use a set to avoid re-checking the same starts
    checked_starts = set()
    
    # Strategy: Try random starts, but also try starts near known good regions
    # Focus on ranges where phi values tend to be more distinct
    
    while time.time() < DEADLINE - 5.0:
        # Choose a random start, avoid duplicates
        s = random.randint(1, 10**12 - 200)
        if s in checked_starts:
            continue
        checked_starts.add(s)
        
        # Check this start
        k = check_start(s, DEADLINE - 5.0)
        
        if k > best_k:
            best_k = k
            best_s = s
            
        # If we found a very long run, we might stop or continue?
        # Max k is 200. If we hit 200, we can stop.
        if best_k == 200:
            break
        
        # If we are running out of time, break
        if time.time() > DEADLINE - 5.0:
            break

    return {'start': best_s, 'length': best_k}

# EVOLVE-BLOCK-END
