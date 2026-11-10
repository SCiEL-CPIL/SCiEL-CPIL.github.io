import random
import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for overhead/termination
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Efficient phi function
    def phi(x):
        if x == 1:
            return 1
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    best_k = 1
    best_s = 1
    max_k = 200
    max_s = 10**12 - max_k + 1
    
    # 3. Search
    # We will try random samples.
    # Also, we can try a few deterministic high-value spots if random fails (unlikely).
    
    while time.time() < deadline:
        # Pick a random start
        s = random.randint(1, max_s)
        
        seen = set()
        k = 0
        # Check consecutive integers
        for i in range(max_k):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
            
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == max_k:
                # Found maximum possible length
                break
                
    return {'start': best_s, 'length': best_k}
