import math
import random

def score_set(U):
    if not U or U[0] != 0:
        return 0
    U = sorted(U)
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute |U+U|
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sum_set.add(U[i] + U[j])
            
    # Compute |U-U|
    diff_set = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diff_set.add(U[i] - U[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    best_score = 0
    best_set = [0]
    
    # Parameters for search
    max_iterations = 5000
    n_start = 3
    max_val_start = 50
    
    # Try different sizes and ranges
    for n in range(3, 15):
        for max_val in range(10, 200, 10):
            # Initialize with a random set or greedy
            # Greedy: start with 0, add elements that maximize score incrementally? 
            # Or just random restarts.
            
            for restart in range(20):
                # Random initialization
                # Ensure 0 is in the set
                # Pick n-1 unique values from 1..max_val
                if max_val < n:
                    continue
                U = [0] + random.sample(range(1, max_val + 1), n - 1)
                U.sort()
                
                current_score = score_set(U)
                if current_score > best_score:
                    best_score = current_score
                    best_set = U[:]
                    
                # Local Search
                stuck = 0
                while stuck < 500:
                    # Pick a random index to modify
                    idx = random.randint(1, n - 1) # Don't change 0
                    old_val = U[idx]
                    
                    # Try changing to a nearby value or random value in range
                    # Strategy 1: Small perturbation
                    delta = random.randint(-5, 5)
                    new_val = old_val + delta
                    if new_val < 1 or new_val > max_val:
                        # Try random value
                        new_val = random.randint(1, max_val)
                    
                    if new_val in U:
                        continue
                        
                    U[idx] = new_val
                    U.sort()
                    
                    new_score = score_set(U)
                    
                    if new_score > current_score:
                        current_score = new_score
                        stuck = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_set = U[:]
                    else:
                        # Revert
                        U[idx] = old_val
                        U.sort()
                        stuck += 1
                        # If stuck too long, break
                        if stuck > 100:
                            break
                            
    return best_set
