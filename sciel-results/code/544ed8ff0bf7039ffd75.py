import random
import time
import math

def construct():
    # Miller-Rabin primality test for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        
        # Write n-1 as d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        
        # Bases sufficient for 64-bit
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Rho factorization
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
                d = math.gcd(abs(x - y), n)
            
            if d != n:
                return d

    def factorize(n):
        if n == 1:
            return []
        factors = []
        
        def _factorize(m):
            if m == 1:
                return
            if is_prime(m):
                factors.append(m)
                return
            
            d = pollard_rho(m)
            _factorize(d)
            _factorize(m // d)
        
        _factorize(n)
        return factors

    def phi(n):
        if n == 1:
            return 1
        factors = factorize(n)
        result = n
        for p in set(factors):
            result -= result // p
        return result

    best_k = 1
    best_s = 1
    
    start_time = time.time()
    time_limit = 115
    
    # Phase 1: Small exhaustive search using fast phi (still fast enough for 10^5)
    # But actually, random search with fast phi is better for large k.
    # Let's just do random search immediately, but maybe start with a few small ones to establish a baseline.
    
    # Try a few small starts to ensure we have a valid answer immediately
    for s in [1, 10, 100, 1000, 10000]:
        seen = set()
        k = 0
        while k < 200:
            v = phi(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
        if time.time() - start_time > time_limit:
            break

    # Phase 2: Intensive Random Search
    # Bias towards larger numbers? 
    # phi(n) for n ~ 10^12 is ~ 10^12. Collisions are less likely in large ranges?
    # Actually, the distribution of phi(n) is complex.
    # Let's sample log-uniformly.
    
    while time.time() - start_time < time_limit:
        # Pick random start
        # Avoid very small numbers where collisions are frequent?
        # Let's try uniform in [10^4, 10^12]
        log_s = random.uniform(4, 12)
        s = int(10 ** log_s)
        
        if s < 1:
            s = 1
        if s + 200 > 10**12:
            s = 10**12 - 200
        
        seen = set()
        k = 0
        while k < 200:
            v = phi(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
            
            # Optional: If we found a good one, try to extend or search nearby?
            # Nearby search might be too slow if phi is expensive.
            # But we can try a few shifts.
            # Let's just stick to random for now, as phi is fast.

    return {'start': best_s, 'length': best_k}
