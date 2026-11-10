import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def count_nonzero(poly):
        return int(np.count_nonzero(poly))
    
    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)
    
    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)
    
    best_score = float('inf')
    best_poly = [1] * 12  # baseline
    
    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if len(coeffs) != 12:
            return
        if sum(1 for c in coeffs if c != 0) != 12:
            return
        if any(c < -64 or c > 64 for c in coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)
    
    # Try baseline
    try_poly([1] * 12)
    
    # Structured candidates based on known sparse constructions
    # Try polynomials where coefficients create systematic cancellations
    structured_candidates = []
    
    # Alternating signs
    structured_candidates.append([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    # Pattern: 1,1,-1,-1 repeating
    structured_candidates.append([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
    # Pattern with larger coefficients
    structured_candidates.append([1, 2, 1, -1, -2, -1, 1, 2, 1, -1, -2, -1])
    # Try: 1, -2, 1, 0, 1, -2, 1, 0, 1, -2, 1, 0 (but need 12 nonzero)
    structured_candidates.append([1, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 1])
    # Try symmetric patterns
    structured_candidates.append([1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1])
    structured_candidates.append([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6])
    structured_candidates.append([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
    structured_candidates.append([1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1])
    
    for s in structured_candidates:
        if len(s) == 12 and sum(1 for c in s if c != 0) == 12:
            if all(-64 <= c <= 64 for c in s):
                try_poly(s)
    
    # Now do targeted local search
    random.seed(42)
    
    # Key insight: try to find polynomials where the square has many zeros
    # Use simulated annealing approach with better neighborhood moves
    
    def local_search(start, max_iters=5000):
        curr = list(start)
        curr_score = evaluate(curr)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try changing one coefficient
            pos = random.randint(0, 11)
            old_val = curr[pos]
            
            # Try a range of values around current
            step = random.choice([1, 2, 4, 8, 16])
            new_val = old_val + random.choice([-step, step])
            
            if -64 <= new_val <= 64 and new_val != 0:
                curr[pos] = new_val
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    try_poly(curr)
                else:
                    curr[pos] = old_val
            
            # Try swapping two coefficients
            if random.random() < 0.3:
                i, j = random.sample(range(12), 2)
                a, b = curr[i], curr[j]
                curr[i], curr[j] = b, a
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    try_poly(curr)
                else:
                    curr[i], curr[j] = a, b
            
            if not improved and random.random() < 0.05:
                # Occasional random perturbation
                for _ in range(random.randint(1, 3)):
                    p = random.randint(0, 11)
                    curr[p] = random.randint(-64, 64)
                    if curr[p] == 0:
                        curr[p] = 1
        
        return curr, curr_score
    
    # Multiple restarts from different starting points
    for restart in range(300):
        if time.time() > deadline:
            break
        
        if restart == 0:
            start = list(best_poly)
        elif random.random() < 0.4:
            start = list(best_poly)
            # Perturb slightly
            for _ in range(random.randint(1, 4)):
                p = random.randint(0, 11)
                start[p] = max(-64, min(64, start[p] + random.randint(-10, 10)))
                if start[p] == 0:
                    start[p] = 1
        else:
            # Random start with structure
            pattern = random.choice([
                lambda i: random.choice([-1, 1]) * random.randint(1, 8),
                lambda i: random.choice([-1, 1]) * (1 + (i % 3)),
                lambda i: random.choice([-1, 1]) * random.randint(1, 16),
            ])
            start = [pattern(i) for i in range(12)]
            # Ensure all nonzero
            for i in range(12):
                if start[i] == 0:
                    start[i] = 1
        
        # Ensure valid
        start = [max(-64, min(64, max(1, min(64, abs(c)) * (1 if c >= 0 else -1)))) for c in start]
        
        result, score = local_search(start, max_iters=2000)
        if score < best_score:
            best_score = score
            best_poly = result
    
    # Final refinement: try all single-coefficient changes on best
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val
    
    return best_poly
