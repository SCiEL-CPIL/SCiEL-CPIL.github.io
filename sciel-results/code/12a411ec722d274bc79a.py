import time
import math

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
    return factors

def get_divisors(factors):
    divisors = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    return divisors

def construct():
    deadline = time.time() + 50.0
    answers = []
    
    for n in range(3, 257):
        found = None
        x_min = n // 4 + 1
        
        for x in range(x_min, x_min + 100): # Limit x search
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # Strategy: Check if iterating y is faster
            # y range: [max(x+1, b//a + 1), 2*b//a]
            y_lo = max(x + 1, b // a + 1)
            y_hi = 2 * b // a
            
            # If range is small, iterate y
            if y_hi - y_lo < 1000:
                for y in range(y_lo, y_hi + 1):
                    num = a * y - b
                    if num <= 0:
                        continue
                    den = b * y
                    if den % num != 0:
                        continue
                    z = den // num
                    if z <= y:
                        continue
                    if z > 10**12:
                        continue
                    if x != y and y != z and x != z:
                        found = [x, y, z]
                        break
            else:
                # Use divisor method
                # Find divisors u of b^2 such that:
                # 1. u < b
                # 2. u > a*x - b
                # 3. u % a == (-b) % a
                # 4. y = (u+b)/a is integer (implied by 3)
                # 5. z = (b^2/u + b)/a is integer (implied by 3 and u|b^2)
                
                # Factor b^2
                # Since b = n*x, and n, x are small, b is small.
                # We can just factor b and double exponents.
                factors_b = factorize(b)
                factors_b2 = {p: 2*e for p, e in factors_b.items()}
                divs = get_divisors(factors_b2)
                
                target_mod = (-b) % a
                lower_bound = a * x - b
                
                for u in divs:
                    if u >= b:
                        continue
                    if u <= lower_bound:
                        continue
                    if u % a != target_mod:
                        continue
                    
                    y = (u + b) // a
                    v = b * b // u
                    z = (v + b) // a
                    
                    # Check distinctness and bounds
                    if x == y or x == z or y == z:
                        continue
                    if y > 10**12 or z > 10**12:
                        continue
                    if y < 1 or z < 1:
                        continue
                    
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        if time.time() > deadline:
            while len(answers) < 254:
                answers.append(None)
            return answers
        
        answers.append(found)
        
    return answers
