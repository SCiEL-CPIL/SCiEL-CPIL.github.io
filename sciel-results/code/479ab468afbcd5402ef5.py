from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    n = 30
    # LCM of 1..30
    lcm_val = 1
    for i in range(1, n+1):
        lcm_val = lcm(lcm_val, i)
    
    # Convert each 1/i to an integer: weight[i] = lcm_val // i
    weights = [lcm_val // (i+1) for i in range(n)]
    
    def compute_sum_int(coeffs):
        """Return the numerator of the sum when expressed with denominator lcm_val"""
        total = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                total += c * weights[i]
        return total
    
    # Seed value
    seed_coeffs = [0]*28 + [1, -1]
    seed_num = compute_sum_int(seed_coeffs)
    best_abs = abs(seed_num)
    best_coeffs = seed_coeffs[:]
    
    import random
    import time
    
    start = time.time()
    
    for trial in range(500):
        if time.time() - start > 50:
            break
            
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Local search
        improved = True
        while improved:
            improved = False
            current_num = compute_sum_int(coeffs)
            if current_num == 0:
                # Perturb
                for i in range(n):
                    old = coeffs[i]
                    coeffs[i] = random.choice([-1, 0, 1])
                    if compute_sum_int(coeffs) != 0:
                        break
                    coeffs[i] = old
                current_num = compute_sum_int(coeffs)
            
            for i in range(n):
                old = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    coeffs[i] = new_val
                    new_num = compute_sum_int(coeffs)
                    if new_num != 0 and abs(new_num) < abs(current_num):
                        current_num = new_num
                        improved = True
                        break
                coeffs[i] = old
        
        current_num = compute_sum_int(coeffs)
        if current_num != 0 and abs(current_num) < best_abs:
            best_abs = abs(current_num)
            best_coeffs = coeffs[:]
    
    return best_coeffs
