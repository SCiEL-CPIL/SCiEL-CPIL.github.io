import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # We search for coefficients c[0..11] for P(x) = sum c[i] x^i
    # Square will have terms at 0..22
    
    def count_nonzero(poly):
        return int(np.count_nonzero(poly))
        
    def evaluate(coeffs):
        # coeffs is a list of 12 integers
        # Use np.convolve for speed
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) != 12:
            return False
        if coeffs[-1] == 0 or coeffs[0] == 0:
            return False
        if not all(isinstance(c, int) for c in coeffs):
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    random.seed(42)

    # Heuristic: Start with structures that might allow cancellation
    # 1. All 1s (Score 23)
    # 2. Alternating 1s (Score 23)
    # 3. Random small integers
    
    initial_candidates = [
        [1]*12,
        [1 if i%2==0 else -1 for i in range(12)],
        [random.randint(-3, 3) for _ in range(12)]
    ]
    # Ensure non-zero
    for c in initial_candidates:
        for i in range(12):
            if c[i] == 0:
                c[i] = 1

    for c in initial_candidates:
        score = evaluate(c)
        if score < best_score:
            best_score = score
            best_poly = c.copy()

    def local_search(curr, curr_score, max_iters):
        nonlocal best_score, best_poly
        
        no_improve = 0
        max_no_improve = 2000
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                return
            
            if no_improve > max_no_improve:
                # Random restart within local search
                curr = [random.randint(-64, 64) for _ in range(12)]
                for i in range(12):
                    if curr[i] == 0:
                        curr[i] = 1
                curr_score = evaluate(curr)
                no_improve = 0
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = curr.copy()
                continue

            idx = random.randint(0, 11)
            old_val = curr[idx]
            
            # Generate candidates
            # Bias towards small integers and values that might cancel
            # We don't know the exact cancellation condition easily, so random is key
            # But we can try +/- 1, +/- 2, random small, random large
            
            candidates = set()
            candidates.add(old_val + 1)
            candidates.add(old_val - 1)
            candidates.add(-old_val)
            
            # Add some random small integers
            for _ in range(3):
                candidates.add(random.randint(-5, 5))
            
            # Add a random large integer
            candidates.add(random.randint(-64, 64))
            
            improved = False
            best_candidate_val = old_val
            best_candidate_score = curr_score
            
            for new_val in candidates:
                if new_val == 0:
                    continue
                if not (-64 <= new_val <= 64):
                    continue
                if new_val == old_val:
                    continue
                
                curr[idx] = new_val
                new_score = evaluate(curr)
                
                if new_score < best_candidate_score:
                    best_candidate_score = new_score
                    best_candidate_val = new_val
            
            if best_candidate_val != old_val:
                curr[idx] = best_candidate_val
                curr_score = best_candidate_score
                no_improve = 0
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = curr.copy()
            else:
                no_improve += 1

    # Run multiple restarts of local search
    num_restarts = 200 # We can do many fast restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random coefficients
        curr = [random.randint(-64, 64) for _ in range(12)]
        for i in range(12):
            if curr[i] == 0:
                curr[i] = 1
        
        curr_score = evaluate(curr)
        if curr_score < best_score:
            best_score = curr_score
            best_poly = curr.copy()
            
        # Run local search
        # Limit iterations per restart to ensure we finish
        # Estimate time per iteration: ~1-2ms?
        # 85s / 1ms = 85000 iterations total.
        # Let's just run until time limit inside local_search
        local_search(curr, curr_score, 100000)

    if best_poly is None:
        best_poly = [1] * 12

    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly
