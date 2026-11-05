"""Solve 1/x + 1/y + 1/z = 4/n for n=3..256 with distinct positive denominators."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4
        
        if x_min > x_max:
            answers.append(None)
            continue
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            # (ay-b)(az-b) = b^2, let u = ay-b, v = az-b
            # u*v = b^2, u < v, u > 4x^2 - 2nx
            # u ≡ -b (mod a), so u = -b + k*a
            lower_bound = 4 * x * x - 2 * n * x
            b2 = b * b
            
            k_min = max((lower_bound + b) // a + 1, b // a + 1)
            k_max = (2 * b - 1) // a
            
            if k_min > k_max:
                continue
            
            for k in range(k_min, k_max + 1):
                u = -b + k * a
                if u <= 0 or u >= b:
                    continue
                if b2 % u != 0:
                    continue
                v = b2 // u
                if v <= u:
                    continue
                if (v + b) % a != 0:
                    continue
                y = (u + b) // a
                z = (v + b) // a
                if x == y or y == z or x == z:
                    continue
                if y <= x or z <= y:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
