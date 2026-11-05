"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Search for minimal Egyptian fraction with denominators >= 5
    found = None
    
    # Try 3-term solutions: 1/a + 1/b + 1/c = 1, 5 <= a <= b <= c
    if time.time() < deadline:
        for a in range(5, 200):
            if time.time() > deadline:
                break
            # 1/b + 1/c = (a-1)/a => c = a*b / ((a-1)*b - a)
            # For c >= b >= a >= 5, and c integer
            for b in range(a, 10000):
                if time.time() > deadline:
                    break
                denom = (a - 1) * b - a
                if denom <= 0:
                    continue
                num = a * b
                if num % denom != 0:
                    continue
                c = num // denom
                if c < b:
                    continue
                if c > 10000000:
                    break
                # Verify: b*c + a*c + a*b == a*b*c
                if b * c + a * c + a * b == a * b * c:
                    found = [a, b, c]
                    break
            if found:
                break
    
    if not found and time.time() < deadline:
        # Try 4-term solutions: 1/a + 1/b + 1/c + 1/d = 1
        for a in range(5, 30):
            if time.time() > deadline:
                break
            for b in range(a, 100):
                if time.time() > deadline:
                    break
                # 1/c + 1/d = (a-1)/a - 1/b = ((a-1)*b - a) / (a*b)
                num_cd = (a - 1) * b - a
                den_cd = a * b
                if num_cd <= 0:
                    continue
                # c = den_cd * b' / (num_cd * b' - den_cd) ... 
                # 1/c + 1/d = num_cd/den_cd => d = den_cd*c / (num_cd*c - den_cd)
                # For d >= c >= b, iterate c
                for c in range(b, 10000):
                    if time.time() > deadline:
                        break
                    denom2 = num_cd * c - den_cd
                    if denom2 <= 0:
                        continue
                    num2 = den_cd * c
                    if num2 % denom2 != 0:
                        continue
                    d = num2 // denom2
                    if d < c:
                        continue
                    if d > 10000000:
                        break
                    # Verify
                    if b * c * d + a * c * d + a * b * d + a * b * c == a * b * c * d:
                        found = [a, b, c, d]
                        break
                if found:
                    break
            if found:
                break
    
    if found:
        if 1 <= len(found) <= 256:
            if all(5 <= x <= 10000000 for x in found):
                if len(set(found)) == len(found):
                    return found
    
    # Fallback to Greedy Algorithm
    num, den = 1, 1
    used = set()
    result = []
    
    while num > 0:
        if time.time() > deadline:
            break
            
        min_d = (den + num - 1) // num
        if min_d < 5:
            min_d = 5
            
        d = min_d
        while d in used:
            d += 1
            
        if d > 10000000:
            break
            
        result.append(d)
        used.add(d)
        
        num = num * d - den
        den = den * d
        
        g = math.gcd(num, den)
        num //= g
        den //= g
        
    if num == 0 and len(result) > 0 and len(result) <= 256:
        return result
        
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
