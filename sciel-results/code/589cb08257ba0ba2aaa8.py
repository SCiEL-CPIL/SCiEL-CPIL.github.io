import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute smallest prime factor for fast factorization up to ~50000
    max_val = 50000
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize_fast(num):
        if num <= 1:
            return {}
        factors = {}
        while num > 1:
            p = spf[num]
            count = 0
            while num % p == 0:
                num //= p
                count += 1
            factors[p] = count
        return factors
    
    def get_divisors_from_factors(factors):
        divisors = [1]
        for p, exp in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors
    
    def combine_factors(f1, f2):
        result = dict(f1)
        for p, e in f2.items():
            result[p] = result.get(p, 0) + e
        return result
    
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        n_factors = factorize_fast(n)
        
        min_x = n // 4 + 1
        max_x = (3 * n) // 4
        
        for x in range(min_x, max_x + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            x_factors = factorize_fast(x)
            b_factors = combine_factors(n_factors, x_factors)
            b2_factors = {p: 2 * e for p, e in b_factors.items()}
            
            divisors = get_divisors_from_factors(b2_factors)
            
            for u in divisors:
                if u >= B:
                    continue
                if (u + B) % A != 0:
                    continue
                
                v = B2 // u
                if v <= u:
                    continue
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                if y > x and z > y and z <= 1000000000000:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
