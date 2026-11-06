import math
import time

def get_divisors(n):
    if n < 2:
        return [1]
    factors = {}
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1 if d == 2 else 2
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    
    divisors = [1]
    for p, exp in factors.items():
        new_divs = []
        for d in divisors:
            for e in range(exp + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs
    return sorted(divisors)

def compute_tau_perp(n):
    if n < 2:
        return 0
    divs = get_divisors(n)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def construct():
    deadline = time.time() + 135
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # 15 primes square-free
    n15 = 1
    for p in primes:
        n15 *= p
    best_n = n15
    best_score = compute_tau_perp(n15)
    
    candidates = {n15}
    
    # Try exponent variations
    for k in range(12, 16):
        if time.time() >= deadline:
            break
        base = 1
        for p in primes[:k]:
            base *= p
        if base > 10**18:
            continue
        
        for a in range(2, 30):
            if time.time() >= deadline:
                break
            nv = base * (2**(a-1))
            if nv > 10**18:
                break
            candidates.add(nv)
            for b in range(2, 20):
                nv2 = nv * (3**(b-1))
                if nv2 > 10**18:
                    break
                candidates.add(nv2)
                for c in range(2, 15):
                    nv3 = nv2 * (5**(c-1))
                    if nv3 > 10**18:
                        break
                    candidates.add(nv3)
    
    # Evaluate
    for n in sorted(candidates):
        if time.time() >= deadline:
            break
        if n < 2 or n > 10**18:
            continue
        s = compute_tau_perp(n)
        if s > best_score:
            best_score = s
            best_n = n
    
    return best_n
