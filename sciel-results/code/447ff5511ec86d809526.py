import math
import random
import time

def score_set(U):
    """
    Calculate the score for a set U of nonnegative integers containing 0.
    Score = 1 + log(|U-U|/|U+U|) / log(2 * max(U) + 1)
    """
    if not U or 0 not in U:
        return 0.0
    
    U_list = list(set(U)) # Ensure unique
    if len(U_list) != len(U):
        # If input had duplicates, we treat it as a set
        pass
        
    if not U_list:
        return 0.0
        
    max_u = max(U_list)
    if max_u == 0:
        # U = {0}
        # U-U = {0}, U+U = {0}
        # Ratio 1. log(1)=0. Score 1.
        return 1.0

    diffs = set()
    sums = set()
    for a in U_list:
        for b in U_list:
            diffs.add(a - b)
            sums.add(a + b)
            
    num_diffs = len(diffs)
    num_sums = len(sums)
    
    if num_sums == 0:
        return 0.0
        
    ratio = num_diffs / num_sums
    denom = math.log(2 * max_u + 1)
    
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def local_search(U, deadline):
    """
    Perform local search to improve the set U.
    """
    current_best = list(set(U))
    current_best.sort()
    best_score = score_set(current_best)
    
    # If 0 is not in set, add it and remove the largest to keep size?
    # The problem states U must contain 0.
    if 0 not in current_best:
        current_best.remove(max(current_best))
        current_best.append(0)
        current_best.sort()
        best_score = score_set(current_best)
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current_best)
        
        # Define search range for new values
        # Max value in current set
        max_val = max(current_best)
        # We might want to expand the range slightly
        search_max = max_val + 10 
        # Or maybe up to 100?
        # Let's try a dynamic range.
        
        for i in range(n):
            old_val = current_best[i]
            
            # Try replacing old_val with new_val
            # Range: 0 to search_max
            # Skip if new_val is already in the set (and not the one we're replacing)
            current_set = set(current_best)
            current_set.remove(old_val)
            
            for new_val in range(0, search_max + 1):
                if new_val in current_set:
                    continue
                
                # Construct new set
                new_set = list(current_set)
                new_set.append(new_val)
                new_set.sort()
                
                # Calculate score
                s = score_set(new_set)
                
                if s > best_score:
                    best_score = s
                    current_best = new_set
                    improved = True
                    # Update max_val for this iteration?
                    # It's safer to just let the outer loop handle it,
                    # or update search_max if we grew.
                    # But we removed one and added one, so size is constant.
                    # The max value might have changed.
                    # We'll just re-calculate search_max at the start of each i-loop?
                    # No, that's expensive.
                    # Let's just use a fixed generous range for the current round.
                    # Or break and restart the i-loop?
                    # Breaking and restarting ensures we always use updated max.
                    # Let's break inner and restart outer.
                    break
            
            if improved:
                # Restart the element iteration to account for changes
                break
            
    return current_best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Initial seeds
    seeds = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 2, 4, 8, 16, 32, 64],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
    ]
    
    # Add some random seeds
    random.seed(42)
    for _ in range(10):
        n = random.randint(3, 8)
        # Generate a random set of size n containing 0
        # Values should be smallish
        max_val = random.randint(10, 50)
        vals = [0]
        while len(vals) < n:
            v = random.randint(1, max_val)
            if v not in vals:
                vals.append(v)
        seeds.append(sorted(vals))
        
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Ensure 0 is in seed
        if 0 not in seed:
            seed = list(seed)
            seed.remove(max(seed))
            seed.append(0)
            seed.sort()
            
        # Run local search
        result_set, result_score = local_search(seed, deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_set = result_set
            
    return best_set
