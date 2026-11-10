import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    LIMIT = 10**12
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            b2 = b * b
            
            u_min = max(1, 4 * x * x - 2 * n * x + 1)
            u_max = b - 1
            
            if u_min > u_max:
                continue
            
            # Find all divisors of b2 in [u_min, u_max]
            # b2 = (n*x)^2
            # We can factor b2 by factoring b and squaring the factors
            
            # Get prime factorization of b
            b_temp = b
            factors = {}
            d = 2
            while d * d <= b_temp:
                while b_temp % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    b_temp //= d
                d += 1
            if b_temp > 1:
                factors[b_temp] = factors.get(b_temp, 0) + 1
            
            # Generate all divisors of b2 from factors of b (each exponent doubled)
            divisors = [1]
            for p, exp in factors.items():
                new_divisors = []
                for d in divisors:
                    for e in range(2 * exp + 1):
                        new_divisors.append(d * (p ** e))
                divisors = new_divisors
            
            # Filter divisors in [u_min, u_max]
            candidates = [u for u in divisors if u_min <= u <= u_max]
            
            # Sort by u to find smallest y first (optional, but might help)
            # Actually, we just need any valid solution
            
            for u in candidates:
                if (u + b) % a != 0:
                    continue
                
                y = (u + b) // a
                
                if y <= 0 or y > LIMIT or y == x:
                    continue
                    
                v = b2 // u
                if (v + b) % a != 0:
                    continue
                    
                z = (v + b) // a
                
                if z <= 0 or z > LIMIT or z == x or z == y:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
