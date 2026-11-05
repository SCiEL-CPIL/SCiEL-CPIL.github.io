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
        if time.time() > deadline:
            while len(answers) < 254:
                answers.append(None)
            return answers
            
        found = None
        x_min = n // 4 + 1
        
        for x in range(x_min, x_min + 300):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            y_lo = max(x + 1, b // a + 1)
            y_hi = 2 * b // a
            
            if y_hi - y_lo < 5000:
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
                # Use divisors of b, not b²
                # (ay-b)(az-b) = b²
                # Let u = ay-b, then u | b² and u ≡ -b (mod a)
                # u = d * e where d*e = b², so d|b and e = b²/d
                # Check d ≡ -b (mod a) or e ≡ -b (mod a)
                divs_b = get_divisors(factorize(b))
                target = (-b) % a
                b2 = b * b
                
                for d in divs_b:
                    e = b2 // d
                    # Check d ≡ target (mod a)
                    if d % a == target and d < b:
                        y = (d + b) // a
                        z = (e + b) // a
                        if y > 0 and z > 0 and x != y and x != z and y != z:
                            if y <= 10**12 and z <= 10**12:
                                found = [x, y, z]
                                break
                    # Check e ≡ target (mod a)  
                    if found is None and e % a == target and e < b:
                        y = (e + b) // a
                        z = (d + b) // a
                        if y > 0 and z > 0 and x != y and x != z and y != z:
                            if y <= 10**12 and z <= 10**12:
                                found = [x, y, z]
                                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
