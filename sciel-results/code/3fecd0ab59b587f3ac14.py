import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    def count_nonzero(poly):
        return np.count_nonzero(poly)
    
    def poly_square(coeffs):
        # coeffs is a numpy array
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    # We assume the 12 non-zero terms are at indices 0..11 for maximum collision potential.
    # If we spread them out, sums are unique, no cancellation.
    # So we fix the support to {0, 1, ..., 11}.
    
    best_score = float('inf')
    best_poly = None
    
    # Initial candidates
    candidates = []
    
    # 1. Baseline
    candidates.append([1]*12)
    
    # 2. Alternating
    candidates.append([1 if i%2==0 else -1 for i in range(12)])
    
    # 3. Random small ints
    random.seed(42)
    for _ in range(5):
        candidates.append([random.randint(-5, 5) for _ in range(12)])
        # Ensure no zeros
        for i in range(12):
            if candidates[-1][i] == 0:
                candidates[-1][i] = 1

    for init_c in candidates:
        if time.time() > deadline:
            break
            
        curr_c = np.array(init_c, dtype=np.int64)
        curr_score = evaluate(curr_c)
        
        if curr_score < best_score:
            best_score = curr_score
            best_poly = curr_c.copy()
            
        # Local Search
        # We try to improve by changing one coefficient at a time.
        # We also try to shift the support? No, keep it simple first.
        
        no_improvement_count = 0
        max_no_improvement = 2000
        step = 0
        
        while time.time() < deadline and no_improvement_count < max_no_improvement:
            step += 1
            # Pick a random index to modify
            idx = random.randint(0, 11)
            # Pick a random new value
            # Bias towards small values
            new_val = random.randint(-10, 10)
            if new_val == 0:
                new_val = 1 # Must be non-zero
            
            if curr_c[idx] == new_val:
                continue
                
            old_val = curr_c[idx]
            curr_c[idx] = new_val
            
            new_score = evaluate(curr_c)
            
            if new_score < curr_score:
                curr_score = new_score
                no_improvement_count = 0
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = curr_c.copy()
                    if best_score <= 12: # Theoretical min for 12 terms is likely higher, but 12 is a good target
                        break
            else:
                # Revert
                curr_c[idx] = old_val
                no_improvement_count += 1
                
        if best_score <= 12:
            break

    # Convert best_poly to list
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)
        
    # The problem asks for a list of coefficients in ascending degree, length <= 41.
    # Our best_poly is length 12 (indices 0-11).
    # We can pad with zeros to length 41 or just return the trimmed list.
    # "length<=41, nonzero final coefficient"
    # If we return [1, 0, ...], the final coefficient is 0? No, "nonzero final coefficient" usually implies the last element of the returned list must be non-zero.
    # If we return length 12, the last element is index 11, which is non-zero.
    # So returning the length 12 list is valid.
    
    return best_poly.tolist()
