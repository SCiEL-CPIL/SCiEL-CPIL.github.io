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
        solved = False
        
        x_min = n // 4 + 1
        max_x_tries = 500
        for x in range(x_min, x_min + max_x_tries):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            k_lo = max(x + 1, b // a + 1)
            k_hi = (2 * b - 1) // a
            
            if k_hi < k_lo:
                continue
            
            if k_hi - k_lo > 200000:
                k_hi = k_lo + 200000
            
            for k in range(k_lo, k_hi + 1):
                d = k * a - b
                if d <= 0:
                    continue
                if (b * b) % d != 0:
                    continue
                e = (b * b) // d
                if (e + b) % a != 0:
                    continue
                z = (e + b) // a
                y = k
                if y <= x or z <= y:
                    continue
                if x > BOUND or y > BOUND or z > BOUND:
                    continue
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    solved = True
                    break
            if solved:
                break
        
        answers[idx] = found
    
    return answers
