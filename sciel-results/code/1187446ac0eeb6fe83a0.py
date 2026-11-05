import time
import math

def _factorize(num):
    factors = {}
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def _gen_divisors(factors):
    divisors = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divisors:
            val = d
            for i in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # x must be at least 1. 
        # 1/x < 4/n => x > n/4.
        # Also x < y < z implies 3/x > 4/n => x < 3n/4.
        # But let's be safe and go up to n.
        x_min = n // 4 + 1
        x_max = n  # Extended bound
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need (ay - b)(az - b) = b^2
            # Let u = ay - b, v = az - b
            # u * v = b^2
            # u must be a divisor of b^2
            # Also u = ay - b => ay = u + b => y = (u + b) / a
            # So (u + b) must be divisible by a.
            # u + b ≡ 0 (mod a) => u ≡ -b (mod a)
            
            # Factorize b
            b_factors = _factorize(b)
            # Generate divisors of b^2
            # Exponents in b^2 are 2*e
            b2_factors = {p: 2*e for p, e in b_factors.items()}
            divisors = _gen_divisors(b2_factors)
            
            # Sort divisors to find u < v (i.e. u < b^2/u => u < b)
            # Actually u and v are symmetric. We just need one pair.
            # We need y, z distinct.
            # y = (u+b)/a, z = (v+b)/a.
            # If u == v, then y == z, not distinct.
            # So we need u != v.
            
            target_mod = (-b) % a
            
            for u in divisors:
                if u > b: # u < v implies u < b^2/u => u^2 < b^2 => u < b
                    break
                
                if u % a != target_mod:
                    continue
                
                v = (b * b) // u
                if u == v:
                    continue
                
                y_num = u + b
                z_num = v + b
                
                if y_num % a != 0 or z_num % a != 0:
                    continue
                
                y = y_num // a
                z = z_num // a
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds
                if z > 1000000000000:
                    continue
                
                # Valid solution found
                # Ensure x, y, z are positive (they are)
                found = [x, y, z]
                break
            
            if found is not None:
                break
        
        answers.append(found)
    
    while len(answers) < 254:
        answers.append(None)
    
    return answers
