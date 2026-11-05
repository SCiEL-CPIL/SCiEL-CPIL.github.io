import math
import time
import random

def construct():
        start_time = time.time()
        DEADLINE = start_time + 130.0
        
        # Sieve primes up to 10^6
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
        
        def phi(x):
            if x == 1: return 1
            n = x
            result = x
            for p in primes:
                if p * p > n:
                    break
                if n % p == 0:
                    while n % p == 0:
                        n //= p
                    result -= result // p
            if n > 1:
                result -= result // n
            return result

        best_k = 1
        best_s = 1
        
        # Random search
        # To ensure we cover the space, we can use multiple random starts.
        # We can also try a few deterministic starts if random fails?
        # No, random is best for global max.
        
        # Optimization: 
        # If we find a collision at s+i, we know phi(s+i) is in seen.
        # We can't easily reuse this.
        
        # Let's just run as many as possible.
        while time.time() < DEADLINE:
            s = random.randint(1, 10**12 - 200)
            seen = set()
            k = 0
            while k < 200:
                val = phi(s + k)
                if val in seen:
                    break
                seen.add(val)
                k += 1
            
            if k > best_k:
                best_k = k
                best_s = s
                # If we found 200, we can stop
                if best_k == 200:
                    break
            
            # Time check
            if time.time() > DEADLINE - 2.0:
                break

        return {'start': best_s, 'length': best_k}
