import math
import time

def count_coprime_pairs(n):
    # Generate divisors
    divisors = [1]
    p = 2
    while p * p <= n:
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            new_divs = []
            for d in divisors:
                for k in range(exp + 1):
                    new_divs.append(d * (p ** k))
            divisors = new_divs
        p += 1 if p == 2 else 2
    if n > 1:
        new_divs = []
        for d in divisors:
            new_divs.append(d)
            new_divs.append(d * n)
        divisors = new_divs
    
    divisors.sort()
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def construct():
    # Try a few candidates
    candidates = []
    
    # 1. Primorial 47#
    primes_47 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n1 = 1
    for p in primes_47:
        n1 *= p
    candidates.append(n1)
    
    # 2. 2^6 * 43#
    primes_43 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    n2_base = 1
    for p in primes_43:
        n2_base *= p
    n2 = n2_base * (2 ** 5) # 2 is already in base, so 2^6 total
    candidates.append(n2)
    
    # 3. Seed
    primes_19 = [2, 3, 5, 7, 11, 13, 17, 19]
    n3 = 1
    for p in primes_19:
        n3 *= p
    candidates.append(n3)
    
    best_n = n3
    best_score = 0
    
    start_time = time.time()
    
    for n in candidates:
        if n > 10**18:
            continue
        score = count_coprime_pairs(n)
        if score > best_score:
            best_score = score
            best_n = n
        if time.time() - start_time > 110:
            break
            
    return best_n
