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
    
    # 2. Optimized phi calculation with early termination and caching
    phi_cache = {}
    def phi(x):
        if x in phi_cache:
            return phi_cache[x]
        if x == 1: 
            phi_cache[x] = 1
            return 1
        n = x
        result = x
        # Trial division by small primes
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        phi_cache[x] = result
        return result
    
    best_k = 1
    best_s = 1
    
    # 3. Deterministic Search near 10^12
    # High probability of finding k=200 here due to low collision probability
    base = 10**12
    # Try offsets to find a valid start
    # s = base - offset - 199
    # We need s >= 1
    offsets = [0, 100, 500, 1000, 5000, 10000, 50000, 100000, 1000000]
    
    for off in offsets:
        if time.time() > DEADLINE - 10:
            break
        s = base - off - 199
        if s < 1:
            continue
            
        # Check if phi(s), ..., phi(s+199) are distinct
        seen = set()
        k = 0
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
            
        if k > best_k:
            best_k = k
            best_s = s
            
        if best_k == 200:
            break

    # 4. Fallback Random Search if deterministic didn't find 200
    if best_k < 200:
        checked_starts = set()
        while time.time() < DEADLINE - 5.0:
            # Log-uniform sampling
            log_s = random.uniform(0, 12)
            s = max(1, min(int(10**log_s), 10**12 - 200))
            if s in checked_starts:
                continue
            checked_starts.add(s)
            
            seen = set()
            k = 0
            for i in range(200):
                if time.time() > DEADLINE - 5.0:
                    break
                val = phi(s + i)
                if val in seen:
                    break
                seen.add(val)
                k += 1
                
            if k > best_k:
                best_k = k
                best_s = s
                
            if best_k == 200:
                break

    return {'start': best_s, 'length': best_k}

# EVOLVE-BLOCK-END
