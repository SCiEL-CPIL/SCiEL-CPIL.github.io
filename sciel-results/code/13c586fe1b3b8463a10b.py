import numpy as np
import time
import random
import itertools

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
        # Validate: exactly 12 nonzero terms, entries in [-64,64], length <= 41
        # We'll work with length-12 arrays (support 0..11)
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
    
    # Try some structured candidates
    structured = [
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, 2, 1, -1, -2, -1, 1, 2, 1, -1, -2, -1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # only 6 nonzero, invalid
    ]
    for s in structured:
        if sum(1 for c in s if c != 0) == 12:
            try_poly(s)
    
    # Local search: random restarts with coefficient perturbations
    random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Start from best or random
        if restart == 0 or random.random() < 0.3:
            curr = list(best_poly)
        else:
            # Random 12-term polynomial
            signs = [random.choice([-1, 1]) for _ in range(12)]
            mags = [random.randint(1, 64) for _ in range(12)]
            curr = [s * m for s, m in zip(signs, mags)]
        
        curr_score = evaluate(curr)
        
        # Local search: try changing one coefficient at a time
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all positions
            for pos in range(12):
                if time.time() > deadline:
                    break
                old_val = curr[pos]
                # Try a few values
                for new_val in range(-64, 65):
                    if new_val == old_val:
                        continue
                    if new_val == 0:  # would reduce nonzero count below 12
                        continue
                    curr[pos] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        try_poly(curr)
                    else:
                        curr[pos] = old_val
                curr[pos] = old_val
        
        # Also try swapping pairs
        if time.time() < deadline:
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i+1, 12):
                    if time.time() > deadline:
                        break
                    a, b = curr[i], curr[j]
                    # Try some swaps
                    for new_a in range(-64, 65, 16):
                        if new_a == 0:
                            continue
                        for new_b in range(-64, 65, 16):
                            if new_b == 0:
                                continue
                            curr[i] = new_a
                            curr[j] = new_b
                            new_score = evaluate(curr)
                            if new_score < curr_score:
                                curr_score = new_score
                                improved = True
                                try_poly(curr)
                            else:
                                curr[i] = a
                                curr[j] = b
                    curr[i] = a
                    curr[j] = b
    
    # Return the best polynomial found
    return best_poly
