import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    def elapsed():
        return time.time() - start_time
    
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(10**6)
    
    def phi_factor(n, primes):
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
    
    def find_best_run(start, end, primes, max_k=200):
        n = end - start + 1
        if n < 1:
            return (1, start)
        
        phi_vals = [phi_factor(start + i, primes) for i in range(n)]
        
        best_k = 1
        best_s = start
        seen = set()
        left = 0
        
        for right in range(n):
            while phi_vals[right] in seen:
                seen.remove(phi_vals[left])
                left += 1
            seen.add(phi_vals[right])
            k = right - left + 1
            if k > best_k:
                best_k = k
                best_s = start + left
            if best_k >= max_k:
                break
        
        return (best_k, best_s)
    
    # Baseline search
    best_k, best_s = find_best_run(1, 20000, primes)
    
    # Strategic search points
    test_points = set()
    for exp in range(4, 13):
        base = 10 ** exp
        for frac in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            s = int(base * (1 + frac))
            if 1 <= s and s + 200 <= 10**12:
                test_points.add(s)
    
    random.seed(42)
    for _ in range(100):
        s = random.randint(1, 10**12 - 1000)
        test_points.add(s)
    
    test_points = sorted(test_points)
    
    for s in test_points:
        if elapsed() > DEADLINE:
            break
        if s < 1 or s + 1000 > 10**12:
            continue
        e = min(s + 1000, 10**12)
        k, s_found = find_best_run(s, e, primes)
        if k > best_k:
            best_k = k
            best_s = s_found
        if best_k >= 200:
            break
    
    while elapsed() < DEADLINE - 5:
        s = random.randint(1, 10**12 - 1000)
        e = min(s + 1000, 10**12)
        k, s_found = find_best_run(s, e, primes)
        if k > best_k:
            best_k = k
            best_s = s_found
        if best_k >= 200:
            break
    
    return {'start': best_s, 'length': best_k}
