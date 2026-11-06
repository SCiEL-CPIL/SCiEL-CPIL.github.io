import time
from math import isqrt

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        
        # Try x values using divisor-based method: (ay-b)(az-b) = b^2
        x_limit = min(x_min + 300, 1000)
        for x in range(x_min, x_limit + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            b_sq = b * b
            rem = (-b) % a
            if rem == 0:
                start_u = a
            else:
                start_u = rem
            
            # u < b, u ≡ -b (mod a), u | b^2
            for u in range(start_u, b, a):
                if time.time() > deadline:
                    break
                if b_sq % u != 0:
                    continue
                v = b_sq // u
                if v <= u:
                    continue
                if (v + b) % a != 0:
                    continue
                y = (u + b) // a
                z = (v + b) // a
                if y > x and z > y and z <= BOUND:
                    if x != y and x != z and y != z:
                        lhs = n * (y * z + x * z + x * y)
                        rhs = 4 * x * y * z
                        if lhs == rhs:
                            found = [x, y, z]
                            break
            
            if found:
                break
        
        # Fallback: direct y search for larger x
        if not found and time.time() < deadline:
            for x in range(x_limit + 1, min(x_limit + 500, 5000)):
                if time.time() > deadline:
                    break
                
                a = 4 * x - n
                if a <= 0:
                    continue
                
                b = n * x
                y_lo = b // a + 1
                y_hi = 2 * b // a - 1
                
                if y_lo <= x:
                    y_lo = x + 1
                if y_hi < y_lo:
                    continue
                
                if y_hi - y_lo > 30000:
                    y_hi = y_lo + 30000
                
                for y in range(y_lo, y_hi + 1):
                    if time.time() > deadline:
                        break
                    
                    denom = a * y - b
                    if denom <= 0:
                        continue
                    
                    if (b * y) % denom == 0:
                        z = (b * y) // denom
                        if z > y and z <= BOUND:
                            if x != y and x != z and y != z:
                                lhs = n * (y * z + x * z + x * y)
                                rhs = 4 * x * y * z
                                if lhs == rhs:
                                    found = [x, y, z]
                                    break
                
                if found:
                    break
        
        answers[idx] = found
    
    return answers
