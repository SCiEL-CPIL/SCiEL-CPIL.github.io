import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    
    def count_nonzero_square(coeffs, exponents):
        n = len(coeffs)
        max_deg = 2 * max(exponents)
        square = np.zeros(max_deg + 1, dtype=np.int64)
        for i in range(n):
            ci = coeffs[i]
            ei = exponents[i]
            square[ei + ei] += ci * ci
            for j in range(i + 1, n):
                deg = ei + exponents[j]
                square[deg] += 2 * ci * coeffs[j]
        return int(np.count_nonzero(square))
    
    def make_result(coeffs, exponents):
        max_deg = max(exponents)
        result = [0] * (max_deg + 1)
        for c, e in zip(coeffs, exponents):
            result[e] = c
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result
    
    def validate(coeffs, exponents):
        if len(coeffs) != 12:
            return False
        if any(c == 0 for c in coeffs):
            return False
        if any(abs(c) > 64 for c in coeffs):
            return False
        if max(exponents) * 2 + 1 > 41:
            return False
        return True
    
    best_nonzero = 78
    best_result = None
    random.seed(12345)
    
    # Generate diverse exponent patterns
    exp_patterns = []
    
    # Arithmetic progressions
    for start in range(0, 5):
        for step in [1, 2, 3]:
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 20:
                exp_patterns.append(exps)
    
    # Two clusters of 6
    for gap in range(6, 15):
        exps = list(range(6)) + list(range(gap, gap + 6))
        if max(exps) <= 20:
            exp_patterns.append(exps)
    
    # Mixed patterns
    for _ in range(100):
        exps = sorted(random.sample(range(0, 21), 12))
        exp_patterns.append(exps)
    
    # Evaluate structured patterns with various coefficient assignments
    for exps in exp_patterns:
        if time.time() > deadline:
            break
        if not validate([1] * 12, exps):
            continue
        
        # Try ±1 coefficients with different sign patterns
        for _ in range(200):
            if time.time() > deadline:
                break
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            nz = count_nonzero_square(coeffs, exps)
            if nz < best_nonzero:
                best_nonzero = nz
                best_result = make_result(coeffs, exps)
        
        # Try small integer coefficients
        for _ in range(200):
            if time.time() > deadline:
                break
            coeffs = []
            for _ in range(12):
                c = random.randint(-3, 3)
                if c == 0:
                    c = random.choice([-1, 1])
                coeffs.append(c)
            if not validate(coeffs, exps):
                continue
            nz = count_nonzero_square(coeffs, exps)
            if nz < best_nonzero:
                best_nonzero = nz
                best_result = make_result(coeffs, exps)
    
    # Local search: perturb best solution
    if best_result is not None and time.time() < deadline:
        current_coeffs = []
        current_exps = []
        for i, c in enumerate(best_result):
            if c != 0:
                current_coeffs.append(c)
                current_exps.append(i)
        
        for _ in range(5000):
            if time.time() > deadline:
                break
            new_coeffs = current_coeffs[:]
            idx = random.randint(0, 11)
            delta = random.randint(-3, 3)
            new_val = new_coeffs[idx] + delta
            if new_val == 0 or abs(new_val) > 64:
                continue
            new_coeffs[idx] = new_val
            nz = count_nonzero_square(new_coeffs, current_exps)
            if nz < best_nonzero:
                best_nonzero = nz
                best_result = make_result(new_coeffs, current_exps)
                current_coeffs = new_coeffs[:]
    
    if best_result is None:
        best_result = [1] * 12
    
    if len(best_result) > 41 or best_result[-1] == 0:
        best_result = [1] * 12
    if sum(1 for c in best_result if c != 0) != 12:
        best_result = [1] * 12
    if any(c < -64 or c > 64 for c in best_result):
        best_result = [1] * 12
    
    return best_result
