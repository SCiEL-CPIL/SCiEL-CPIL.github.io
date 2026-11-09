import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
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
    # Enhanced with 2-flip moves for better escape from local optima
    def local_search(c, current_sum):
        nonlocal best_val, best_abs, best_c
        improved = True
        iteration = 0
        max_iterations = 200  # limit iterations to avoid infinite loops
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Single-flip moves
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
            
            # Two-flip moves (only if no single flip improved)
            if not improved:
                for i in range(30):
                    if time.time() - start_time > deadline - 5:
                        break
                    old_i = c[i]
                    for new_i in (-1, 0, 1):
                        if new_i == old_i:
                            continue
                        c[i] = new_i
                        for j in range(i + 1, 30):
                            old_j = c[j]
                            for new_j in (-1, 0, 1):
                                if new_j == old_j:
                                    continue
                                c[j] = new_j
                                new_sum = current_sum + (new_i - old_i) * A[i + 1] + (new_j - old_j) * A[j + 1]
                                if new_sum != 0:
                                    new_abs = abs(new_sum)
                                    if new_abs < best_abs:
                                        best_abs = new_abs
                                        best_val = new_sum
                                        best_c = c[:]
                                        current_sum = new_sum
                                        improved = True
                                c[j] = old_j
                        c[i] = old_i
        
        return current_sum
    
    # Random restarts with local search
    random.seed(42)
    
    # First, do local search from the initial point
    c = best_c[:]
    local_search(c, get_sum(c))
    
    # Check time
    if time.time() - start_time > deadline:
        return best_c
    
    # Generate diverse starting points for random restarts
    def generate_candidate():
        """Generate a candidate with structure that might lead to small sums."""
        strategy = random.random()
        
        if strategy < 0.3:
            # Pairs of adjacent terms with opposite signs
            c = [0] * 30
            num_pairs = random.randint(1, 5)
            used = set()
            for _ in range(num_pairs):
                i = random.randint(0, 28)
                if i not in used and (i+1) not in used:
                    c[i] = 1
                    c[i+1] = -1
                    used.add(i)
                    used.add(i+1)
            # Ensure non-zero sum
            if get_sum(c) == 0:
                c[random.randint(0, 29)] = random.choice([-1, 1])
            return c
        
        elif strategy < 0.5:
            # Sparse random
            c = [0] * 30
            num_nonzero = random.randint(2, 8)
            indices = random.sample(range(30), num_nonzero)
            for i in indices:
                c[i] = random.choice([-1, 1])
            if get_sum(c) == 0:
                c[random.randint(0, 29)] = random.choice([-1, 1])
            return c
        
        elif strategy < 0.7:
            # Dense random
            c = [random.choice([-1, 0, 1]) for _ in range(30)]
            if get_sum(c) == 0:
                c[random.randint(0, 29)] = random.choice([-1, 1])
            return c
        
        elif strategy < 0.85:
            # Block structure: consecutive +1s then consecutive -1s
            c = [0] * 30
            start = random.randint(0, 25)
            len_pos = random.randint(1, 4)
            len_neg = random.randint(1, 4)
            if start + len_pos + len_neg <= 30:
                for k in range(len_pos):
                    c[start + k] = 1
                for k in range(len_neg):
                    c[start + len_pos + k] = -1
            if get_sum(c) == 0:
                c[random.randint(0, 29)] = random.choice([-1, 1])
            return c
        
        else:
            # Near-optimal perturbation: start from best_c and flip a few
            c = best_c[:]
            num_flips = random.randint(1, 3)
            indices = random.sample(range(30), num_flips)
            for i in indices:
                c[i] = random.choice([-1, 0, 1])
            if get_sum(c) == 0:
                c[random.randint(0, 29)] = random.choice([-1, 1])
            return c
    
    # Random restarts
    restart_count = 0
    while time.time() - start_time < deadline:
        restart_count += 1
        
        # Generate a random candidate
        c = generate_candidate()
        s = get_sum(c)
        if s == 0:
            continue
        
        # Do local search
        local_search(c, s)
        
        # Adaptive: if we've been stuck, try more aggressive exploration
        if restart_count % 50 == 0 and best_abs == abs(A[29] - A[30]):
            # Try targeted search around known good structures
            for i in range(1, 30):
                c = [0] * 30
                c[i-1] = 1
                c[i] = -1
                s = get_sum(c)
                if s != 0:
                    local_search(c, s)
                if time.time() - start_time > deadline:
                    break
        
        if time.time() - start_time > deadline:
            break
    
    return best_c
