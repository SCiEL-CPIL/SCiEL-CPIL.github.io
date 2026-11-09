import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin
    
    # Precompute LCM of 1..30 for exact rational arithmetic
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i
    
    # Initial solution: [0]*28 + [1, -1] gives sum 1/29 - 1/30 = 1/(29*30)
    # In integer terms: A[29] - A[30]
    best_val = A[29] - A[30]
    best_abs = abs(best_val)
    best_c = [0] * 30
    best_c[28] = 1
    best_c[29] = -1
    
    def get_sum(c):
        s = 0
        for i in range(30):
            if c[i] != 0:
                s += c[i] * A[i + 1]
        return s
    
    # Local search: try flipping or changing individual coefficients
    def local_search(c, current_sum):
        nonlocal best_val, best_abs, best_c
        improved = True
        while improved:
            improved = False
            for i in range(30):
                old = c[i]
                for new_val in (-1, 0, 1):
                    if new_val == old:
                        continue
                    c[i] = new_val
                    new_sum = current_sum + (new_val - old) * A[i + 1]
                    if new_sum != 0:
                        new_abs = abs(new_sum)
                        if new_abs < best_abs:
                            best_abs = new_abs
                            best_val = new_sum
                            best_c = c[:]
                            current_sum = new_sum
                            improved = True
                    c[i] = old
        return current_sum
    
    # Random restarts with local search
    random.seed(42)
    
    # First, do local search from the initial point
    c = best_c[:]
    local_search(c, get_sum(c))
    
    # Check time
    if time.time() - start_time > deadline:
        return best_c
    
    # Random restarts
    while time.time() - start_time < deadline:
        # Generate a random candidate
        c = [random.choice([-1, 0, 1]) for _ in range(30)]
        s = get_sum(c)
        if s == 0:
            continue
        
        # Do local search
        local_search(c, s)
        
        if time.time() - start_time > deadline:
            break
    
    return best_c
