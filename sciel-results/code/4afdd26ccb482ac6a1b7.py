import math
from math import isqrt

def is_powerful(n):
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count == 1:
                return False
        p += 1 if p == 2 else 2
    if temp > 1:
        # temp is prime, count is 1
        return False
    return True

def generate_powerful(limit):
    powerful = set()
    # Squares
    max_k = isqrt(limit)
    for k in range(1, max_k + 1):
        powerful.add(k * k)
    
    # Non-square powerful numbers: k^2 * m^3
    # m must be such that m^3 <= limit
    max_m = int(limit ** (1/3))
    for m in range(2, max_m + 1):
        m3 = m * m * m
        if m3 > limit:
            break
        # k^2 <= limit / m3
        max_k = isqrt(limit // m3)
        for k in range(1, max_k + 1):
            val = k * k * m3
            if val <= limit:
                powerful.add(val)
    return powerful

def construct():
    # Optimize N
    primes_1_mod_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137]
    best_N = 0
    best_score = 0
    best_exponents = []
    
    # Recursive search for exponents
    def search(idx, current_N, current_score, exponents):
        nonlocal best_N, best_score, best_exponents
        if current_N > 10**18:
            return
        if idx == len(primes_1_mod_4):
            if current_score > best_score:
                best_score = current_score
                best_N = current_N
                best_exponents = exponents[:]
            return
        
        p = primes_1_mod_4[idx]
        # Try exponents 0, 1, 2, ...
        val = current_N
        for e in range(0, 20):
            if val > 10**18:
                break
            new_score = current_score * (e + 1)
            search(idx + 1, val, new_score, exponents + [e])
            val *= p
            
    search(0, 1, 1, [])
    
    if best_N == 0:
        # Fallback
        best_N = 10**16
        best_exponents = [0]*15

    # Generate powerful numbers up to best_N // 2
    limit = best_N // 2
    powerful_set = generate_powerful(limit)
    
    pairs = []
    # We need to check a <= b
    # a is from powerful_set
    # b = best_N - a
    # b must be powerful
    # If b <= limit, b is in powerful_set
    # If b > limit, we check is_powerful(b)
    
    # Optimization: iterate only over a <= best_N // 2
    for a in powerful_set:
        if a > best_N // 2:
            continue
        b = best_N - a
        if b < a:
            continue
        if b <= limit:
            if b in powerful_set:
                pairs.append([a, b])
        else:
            if is_powerful(b):
                pairs.append([a, b])
                
    return {'N': best_N, 'pairs': pairs}
