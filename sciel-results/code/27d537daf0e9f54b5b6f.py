import math
import random
import time

def construct():
    # Deadline guard: 110 seconds to be safe against 120s limit
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate primes up to 10^6 (sqrt of 10^12)
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result

    best_k = 1
    best_s = 1
    
    # We aim for k=200. If found, return immediately.
    # If not found within time, return best found.
    # Given low collision prob, we expect to find k=200 very fast.
    
    max_k_target = 200
    max_s = 10**12 - max_k_target + 1
    
    # Initial fallback: check a small s just in case random fails or is slow?
    # No, random is faster for large k.
    
    while time.time() < deadline:
        # Sample s from the upper range where phi(n) is large
        s = random.randint(1, max_s)
        
        seen = set()
        current_k = 0
        success = True
        
        for i in range(max_k_target):
            val = phi(s + i)
            if val in seen:
                success = False
                break
            seen.add(val)
            current_k += 1
            
        if success:
            return {'start': s, 'length': max_k_target}
        
        if current_k > best_k:
            best_k = current_k
            best_s = s
            
    return {'start': best_s, 'length': best_k}
