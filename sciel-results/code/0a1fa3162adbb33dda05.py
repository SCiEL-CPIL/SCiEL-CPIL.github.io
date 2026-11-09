import time
import itertools
import random

def calculate_score(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    U = list(set(U))
    U.sort()
    
    # Calculate U-U and U+U
    # Using sets for efficiency
    diffs = set()
    sums = set()
    for a in U:
        for b in U:
            diffs.add(a - b)
            sums.add(a + b)
    
    num_diffs = len(diffs)
    num_sums = len(sums)
    
    if num_sums == 0:
        return 0
        
    import math
    ratio = num_diffs / num_sums
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    score = 1 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, leaving margin for 120s limit
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 4, 16, 64],
        [0, 1, 5, 25],
    ]
    
    best_set = candidates[0]
    best_score = calculate_score(best_set)
    
    # Local Search / Hill Climbing
    # We will try to modify the set by adding, removing, or swapping elements.
    # To keep it fast, we limit the range of elements to [0, 200] initially,
    # and maybe expand if we find good structures.
    
    max_val_limit = 200
    
    def neighbors(U):
        U = list(set(U))
        U.sort()
        max_u = max(U) if U else 0
        
        # 1. Add a new element
        for i in range(0, min(max_val_limit, max_u * 2 + 10)):
            if i not in U:
                yield U + [i]
        
        # 2. Remove an element (keep at least 2 elements, and 0)
        if len(U) > 2:
            for i in range(1, len(U)): # Don't remove 0
                new_U = U[:i] + U[i+1:]
                yield new_U
        
        # 3. Swap an element (except 0)
        for i in range(1, len(U)):
            old_val = U[i]
            for new_val in range(0, min(max_val_limit, max_u * 2 + 10)):
                if new_val != old_val and new_val not in U:
                    new_U = U.copy()
                    new_U[i] = new_val
                    new_U.sort()
                    yield new_U

    # Run hill climbing for a few iterations or until time runs out
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_best = best_set
        current_score = best_score
        
        # Get neighbors
        # To avoid generating too many, we might sample if the space is huge,
        # but for small sets (size < 10) and max_val < 200, it's manageable.
        
        # Optimization: Only try neighbors that are likely better?
        # Just iterate.
        
        for neighbor in neighbors(current_best):
            if time.time() > deadline:
                break
            score = calculate_score(neighbor)
            if score > current_score:
                current_score = score
                current_best = neighbor
                improved = True
                # Continue searching from this new best? 
                # Or restart? 
                # Standard hill climbing: move to the best neighbor or first improving neighbor.
                # Let's move to the first improving neighbor for speed, 
                # or the best neighbor for quality.
                # Best neighbor is better but slower.
                # Given the small size, let's find the best neighbor in this iteration.
                # Actually, the loop above updates current_best immediately.
                # This is "steepest ascent" if we check all, or "first ascent" if we break.
                # Let's check all neighbors to find the steepest ascent.
        
        if improved:
            best_set = current_best
            best_score = current_score
        else:
            # If no improvement, try a random restart or different seed?
            # Or break.
            # Let's try a few random restarts if time permits.
            break
            
    return best_set
