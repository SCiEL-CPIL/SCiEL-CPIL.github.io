import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    def get_divisors_of_square(b):
        n = b
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        divs = [1]
        for p, e in factors.items():
            exp = 2 * e
            current_divs = []
            for d in divs:
                val = d
                for _ in range(exp + 1):
                    current_divs.append(val)
                    val *= p
            divs = current_divs
        return divs

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        x_min = n // 4 + 1
        # x < 3n/4 => x <= (3n-1)//4
        x_max = (3 * n - 1) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            divs = get_divisors_of_square(b)
            
            for k in divs:
                if k < 0:
                    continue
                if k >= b:
                    continue
                    
                if (k + b) % a != 0:
                    continue
                
                y = (k + b) // a
                if y <= x:
                    continue
                
                term = b * b // k
                if (term + b) % a != 0:
                    continue
                
                z = (term + b) // a
                
                if z <= y:
                    continue
                
                if z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
