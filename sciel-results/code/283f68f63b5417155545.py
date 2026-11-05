import time
from functools import lru_cache

@lru_cache(maxsize=None)
def factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return tuple(sorted(factors.items()))

def get_divisors_of_square(factors):
    divisors = [1]
    for p, e in factors:
        max_exp = 2 * e
        new_divs = []
        for d in divisors:
            p_pow = 1
            for k in range(max_exp + 1):
                new_divs.append(d * p_pow)
                p_pow *= p
        divisors = new_divs
    return divisors

def construct():
    deadline = time.time() + 85.0
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            while len(answers) < 254:
                answers.append(None)
            return answers
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 2):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            lower_bound = a * x - b
            if lower_bound >= b - 1:
                continue
                
            fac_b = factorize(b)
            divs_b2 = get_divisors_of_square(fac_b)
            
            target_mod = (-b) % a
            
            for u in divs_b2:
                if u >= b:
                    continue
                if u <= lower_bound:
                    continue
                if u % a != target_mod:
                    continue
                
                num_y = u + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                
                if y <= x:
                    continue
                    
                v = (b * b) // u
                num_z = v + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                
                if z <= y or z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
