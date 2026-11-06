import math
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Precompute smallest prime factor up to 100000
    MAX_VAL = 100000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        factors = {}
        while num > 1:
            p = spf[num]
            cnt = 0
            while num % p == 0:
                num //= p
                cnt += 1
            factors[p] = cnt
        return factors
    
    def get_divisors_from_factors(factors):
        divs = [1]
        for p, e in factors.items():
            max_e = 2 * e
            new_divs = []
            for d in divs:
                for k in range(max_e + 1):
                    new_divs.append(d * (p ** k))
            divs = new_divs
        return divs
    
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        min_x = n // 4 + 1
        max_x = (3 * n) // 4
        
        for x in range(min_x, max_x + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
            
            # Factorize B and get divisors of B^2
            B_factors = factorize(B)
            divs = get_divisors_from_factors(B_factors)
            
            # For each divisor u of B^2 where u < B:
            # Check (u + B) % A == 0
            # y = (u + B) // A must be > x
            # v = B^2 // u
            # z = (v + B) // A must be > y
            # All distinct
            
            B_sq = B * B
            for u in divs:
                if u >= B:
                    continue
                if (u + B) % A != 0:
                    continue
                y = (u + B) // A
                if y <= x:
                    continue
                v = B_sq // u
                if (v + B) % A != 0:
                    continue
                z = (v + B) // A
                if z <= y:
                    continue
                # All three are distinct since x < y < z
                found = [x, y, z]
                break
        
        answers.append(found)
    
    return answers
