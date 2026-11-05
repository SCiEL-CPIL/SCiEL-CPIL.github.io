import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    n = 30
    
    # Compute L = lcm(1, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Integer weights: w[i] = L // i for i = 1..n
    w = [0] * (n + 1)
    for i in range(1, n + 1):
        w[i] = L // i
    
    # We want to find c in {-1,0,1}^30 such that |sum(c[i]*w[i])| is minimized and nonzero
    # The minimum possible nonzero absolute value is 1 (if achievable), giving score = L
    
    best_abs = None
    best_coeffs = None
    
    # Local search: start from random configs, flip/move coefficients to reduce |sum|
    def evaluate(coeffs):
        s = 0
        for i in range(n):
            s += coeffs[i] * w[i + 1]
        return s
    
    def local_search(coeffs, time_limit):
        nonlocal best_abs, best_coeffs
        cur_coeffs = coeffs[:]
        cur_sum = evaluate(cur_coeffs)
        
        if cur_sum != 0 and (best_abs is None or abs(cur_sum) < best_abs):
            best_abs = abs(cur_sum)
            best_coeffs = cur_coeffs[:]
        
        while time.time() - start_time < time_limit:
            improved = False
            # Try flipping each coefficient
            for i in range(n):
                old_c = cur_coeffs[i]
                # Try the other two values
                for new_c in (-1, 0, 1):
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * w[i + 1]
                    new_sum = cur_sum + delta
                    if new_sum == 0:
                        continue
                    if best_abs is None or abs(new_sum) < best_abs:
                        best_abs = abs(new_sum)
                        best_coeffs = cur_coeffs[:]
                        cur_coeffs[i] = new_c
                        cur_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Stuck in local minimum, try a random perturbation
                for _ in range(3):
                    i = random.randint(0, n - 1)
                    cur_coeffs[i] = random.choice([-1, 0, 1])
                cur_sum = evaluate(cur_coeffs)
                if cur_sum != 0 and (best_abs is None or abs(cur_sum) < best_abs):
                    best_abs = abs(cur_sum)
                    best_coeffs = cur_coeffs[:]
        
        return cur_coeffs
    
    # Multi-start local search
    random.seed(42)
    
    # Start with several random initial configurations
    num_starts = 50
    time_per_start = (deadline - (time.time() - start_time)) / num_starts
    
    for start_idx in range(num_starts):
        if time.time() - start_time > deadline:
            break
        
        # Random initial configuration
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Ensure it's not the zero configuration
        if all(c == 0 for c in coeffs):
            coeffs[0] = 1
        
        local_search(coeffs, deadline - 2.0)
        
        # Early termination if we found the optimal
        if best_abs == 1:
            break
    
    if best_coeffs is not None:
        return best_coeffs
    
    # Fallback: simple nonzero solution
    fallback = [0] * n
    fallback[0] = 1
    return fallback
