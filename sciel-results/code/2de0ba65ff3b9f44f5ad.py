import numpy as np
import time
import random
from itertools import product

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
    
    # KEY STRUCTURAL CHANGE: Try polynomials with algebraic structure
    # 1. Polynomials that are sums of geometric series
    # 2. Polynomials with periodic coefficient patterns
    # 3. Polynomials related to cyclotomic polynomials
    
    # Try periodic patterns with period 2, 3, 4, 6
    for period in [2, 3, 4, 6]:
        for base in range(-4, 5):
            for amplitude in range(1, 5):
                pattern = []
                for i in range(12):
                    val = base + amplitude * ((i % period) - period // 2)
                    if val == 0:
                        val = 1
                    pattern.append(val)
                if all(-64 <= c <= 64 for c in pattern):
                    try_poly(pattern)
    
    # Try: p(x) = 1 + x + ... + x^{k-1} - x^k - ... - x^{2k-1} type patterns
    for k in range(2, 7):
        for sign in [1, -1]:
            poly = [0] * 12
            for i in range(min(k, 12)):
                poly[i] = sign
            for i in range(k, min(2*k, 12)):
                poly[i] = -sign
            # Fill remaining with small values to ensure 12 nonzero
            for i in range(2*k, 12):
                if poly[i] == 0:
                    poly[i] = sign * (1 + (i % 3))
            if sum(1 for c in poly if c != 0) == 12:
                try_poly(poly)
    
    # Try: p(x) where coefficients are powers of 2 with signs
    for signs in product([1, -1], repeat=12):
        if time.time() > deadline:
            break
        poly = [signs[i] * (2 ** (i % 4)) for i in range(12)]
        if all(-64 <= c <= 64 for c in poly):
            try_poly(poly)
    
    # Try: p(x) = q(x) * r(x) where q and r are sparse
    # q has 2-3 terms, r has 4-6 terms, product has 12 terms
    # This is complex, so try simpler: p(x) with coefficients from small set
    for coeff_set in [
        [-3, -2, -1, 1, 2, 3],
        [-4, -3, -2, -1, 1, 2, 3, 4],
        [-2, -1, 1, 2],
        [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    ]:
        if time.time() > deadline:
            break
        # Sample random assignments from coeff_set
        for _ in range(5000):
            if time.time() > deadline:
                break
            poly = [random.choice(coeff_set) for _ in range(12)]
            try_poly(poly)
    
    # Now do targeted local search with better moves
    random.seed(42)
    
    def local_search(start, max_iters=3000):
        curr = list(start)
        curr_score = evaluate(curr)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try changing one coefficient to a value that might create cancellation
            pos = random.randint(0, 11)
            old_val = curr[pos]
            
            # Try values that are related to other coefficients (for cancellation)
            candidate_vals = set()
            for other_pos in range(12):
                if other_pos != pos:
                    # Try values that would make certain cross-terms cancel
                    candidate_vals.add(curr[other_pos])
                    candidate_vals.add(-curr[other_pos])
                    candidate_vals.add(2 * curr[other_pos])
                    candidate_vals.add(-2 * curr[other_pos])
            
            # Add some random values
            for _ in range(5):
                candidate_vals.add(random.randint(-64, 64))
            
            # Filter valid values
            candidate_vals = [v for v in candidate_vals if -64 <= v <= 64 and v != 0 and v != old_val]
            
            if candidate_vals:
                new_val = random.choice(candidate_vals)
                curr[pos] = new_val
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    try_poly(curr)
                else:
                    curr[pos] = old_val
            
            # Try swapping two coefficients
            if random.random() < 0.2:
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
            
            # Try negating a coefficient
            if random.random() < 0.1:
                pos = random.randint(0, 11)
                old_val = curr[pos]
                new_val = -old_val
                if new_val != 0:
                    curr[pos] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        try_poly(curr)
                    else:
                        curr[pos] = old_val
            
            if not improved and random.random() < 0.03:
                # Occasional larger perturbation
                for _ in range(random.randint(1, 3)):
                    p = random.randint(0, 11)
                    curr[p] = random.randint(-16, 16)
                    if curr[p] == 0:
                        curr[p] = 1
        
        return curr, curr_score
    
    # Multiple restarts
    for restart in range(500):
        if time.time() > deadline:
            break
        
        if restart < 10:
            start = list(best_poly)
        elif random.random() < 0.5:
            start = list(best_poly)
            for _ in range(random.randint(1, 3)):
                p = random.randint(0, 11)
                start[p] = max(-64, min(64, start[p] + random.randint(-8, 8)))
                if start[p] == 0:
                    start[p] = 1
        else:
            # Random start with structure
            pattern_type = random.randint(0, 3)
            if pattern_type == 0:
                start = [random.choice([-1, 1]) * random.randint(1, 6) for _ in range(12)]
            elif pattern_type == 1:
                base = random.randint(1, 4)
                start = [base * random.choice([-1, 1]) * (1 + (i % 3)) for i in range(12)]
            elif pattern_type == 2:
                start = [random.randint(-4, 4) for _ in range(12)]
                for i in range(12):
                    if start[i] == 0:
                        start[i] = random.choice([-1, 1])
            else:
                start = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        
        result, score = local_search(start, max_iters=1500)
        if score < best_score:
            best_score = score
            best_poly = result
    
    # Final refinement
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-32, 33):
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
