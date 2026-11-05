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
    
    # 2. Pollard's rho for factoring large composites
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = gcd(abs(x - y), n)
            if d != n:
                return d
    
    def factorize(n):
        """Return list of prime factors (with multiplicity)"""
        factors = []
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1:
            # Check if n is prime
            is_n_prime = True
            for p in primes:
                if p * p > n:
                    break
                if n % p == 0:
                    is_n_prime = False
                    break
            if is_n_prime:
                factors.append(n)
            else:
                # Use Pollard's rho
                while n > 1:
                    if n == 1:
                        break
                    # Check if n is prime
                    is_n_prime = True
                    for p in primes:
                        if p * p > n:
                            break
                        if n % p == 0:
                            is_n_prime = False
                            break
                    if is_n_prime:
                        factors.append(n)
                        break
                    d = pollard_rho(n)
                    factors.append(d)
                    n //= d
        return factors
    
    # 3. Optimized phi calculation with caching
    phi_cache = {}
    def phi(x):
        if x in phi_cache:
            return phi_cache[x]
        if x == 1: 
            phi_cache[x] = 1
            return 1
        # Get distinct prime factors
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
            # Check if n is prime
            is_n_prime = True
            for p in primes:
                if p * p > n:
                    break
                if n % p == 0:
                    is_n_prime = False
                    break
            if is_n_prime:
                result -= result // n
            else:
                # Factor n using Pollard's rho
                factors = factorize(n)
                # Get distinct factors
                distinct_factors = set(factors)
                for f in distinct_factors:
                    result -= result // f
        phi_cache[x] = result
        return result
    
    best_k = 1
    best_s = 1
    
    # 4. Search Strategy with improved speed
    # We want to find a start s such that phi(s), ..., phi(s+k-1) are distinct.
    # Random sampling of s in [1, 10^12].
    # For each s, we compute phi(s), phi(s+1), ... until a collision or limit k=200 or time out.
    
    # Helper to check a specific start with time limit
    def check_start(s, deadline):
        seen = set()
        k = 0
        limit_k = 200
        while k < limit_k:
            if time.time() > deadline:
                break
            val = phi(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    # Phase 1: Check small ranges for quick baseline
    small_starts = [1, 2, 3, 4, 5, 10, 100, 1000, 10000]
    for s in small_starts:
        if time.time() > DEADLINE - 10:
            break
        k = check_start(s, DEADLINE)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Random Search with improved efficiency
    # Use a set to avoid re-checking the same starts
    checked_starts = set()
    
    # Strategy: Use log-uniform sampling to cover the full range [1, 10^12]
    # This ensures we sample from all orders of magnitude
    while time.time() < DEADLINE - 5.0:
        # Choose a random start using log-uniform distribution
        max_attempts = 50
        found_new = False
        for _ in range(max_attempts):
            # Log-uniform sampling: s = 10^U where U ~ Uniform(0, 12)
            log_s = random.uniform(0, 12)
            s = max(1, min(int(10**log_s), 10**12 - 200))
            if s not in checked_starts:
                checked_starts.add(s)
                found_new = True
                break
        
        if not found_new:
            # Fallback to uniform if log-uniform keeps hitting checked values
            s = random.randint(1, 10**12 - 200)
        
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
