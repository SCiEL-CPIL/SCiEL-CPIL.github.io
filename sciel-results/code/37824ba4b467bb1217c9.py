import time
import math

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        x_start = n // 4 + 1
        x_end = (3 * n + 3) // 4
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            b_sq = b * b
            
            # Generate divisors of b_sq more efficiently by factoring b
            # b = n * x, so factorize n and x separately
            # Then b^2 has squared exponents
            
            # Factorize b
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
            
            # b^2 has exponents doubled
            # Generate all divisors of b^2
            divisors = [1]
            for prime, exp in factors.items():
                exp_sq = exp * 2
                new_divisors = []
                for d in divisors:
                    for e in range(exp_sq + 1):
                        new_divisors.append(d * (prime ** e))
                divisors = new_divisors
            
            # Sort divisors to check smaller ones first (faster to find valid y,z)
            divisors.sort()
            
            for u in divisors:
                if u > b_sq:
                    break
                v = b_sq // u
                if v * u != b_sq:
                    continue
                
                if (u + b) % a != 0:
                    continue
                if (v + b) % a != 0:
                    continue
                
                y = (u + b) // a
                z = (v + b) // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND or x > BOUND:
                    continue
                if x == y or x == z or y == z:
                    continue
                
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
    
        answers[idx] = found
    
    return answers
