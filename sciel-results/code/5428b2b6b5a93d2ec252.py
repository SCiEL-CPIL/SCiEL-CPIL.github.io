import math
import numpy as np
import itertools
import time

def score_set(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Vectorized calculation for speed
    U_arr = np.array(U)
    sums = np.add.outer(U_arr, U_arr).flatten()
    diffs = np.subtract.outer(U_arr, U_arr).flatten()
    
    size_sum = len(np.unique(sums))
    size_diff = len(np.unique(diffs))
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    if ratio <= 0:
        return 0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
        
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # 1. Exhaustive search for small n and small max value
    # n=3, max=20: C(20,2)=190
    # n=4, max=20: C(20,3)=1140
    # n=5, max=20: C(20,4)=4845
    # n=6, max=20: C(20,5)=15504
    # n=7, max=20: C(20,6)=38760
    # n=8, max=20: C(20,7)=77520
    
    # We can go up to n=7, max=30 comfortably.
    # Let's try n up to 6, max up to 40 first.
    
    for n in range(3, 7):
        if n == 6:
            max_val = 30
        else:
            max_val = 40
            
        # Iterate through combinations of 1..max_val of size n-1
        # and prepend 0
        for comb in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                return best_set
                
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    # 2. Local Search / Hill Climbing
    # Start from the best found so far
    current_set = best_set[:]
    current_score = best_score
    
    # Define a neighborhood: change one element by +/- delta
    # We can also try adding or removing elements if it helps, but 
    # the problem asks for a set, and usually larger n helps the ratio 
    # up to a point, but increases max U.
    
    # Let's do a simple hill climbing on the values.
    # Sort the set to keep it canonical
    current_set.sort()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try moving each element
        for i in range(1, len(current_set)): # Don't move 0
            orig_val = current_set[i]
            # Try neighbors
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = orig_val + delta
                if new_val < 0 or new_val >= current_set[-1]: 
                    # We should allow new_val to be anything positive, 
                    # but we must maintain uniqueness and order for simplicity
                    # Actually, we can just try any new_val in a range
                    pass
                
                # Better: try a range of new values
                # Let's just try +/- 1..10
                pass
            
            # Simpler: try all values in [1, max(current_set)+5] that are not in the set
            # This is expensive. Let's stick to small perturbations.
            
            for delta in range(-10, 11):
                if delta == 0:
                    continue
                new_val = orig_val + delta
                if new_val < 1: # 0 is fixed at index 0
                    continue
                if new_val in current_set:
                    continue
                    
                # Create new set
                new_set = current_set[:]
                new_set[i] = new_val
                new_set.sort()
                
                s = score_set(new_set)
                if s > current_score:
                    current_score = s
                    current_set = new_set
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]
                    break # Restart loop with new set

    # 3. Try to add elements
    # Sometimes adding a new element can increase the ratio if it's placed well.
    # But it also increases max U.
    # Let's try adding one element to the best set found.
    
    candidate = best_set[:]
    candidate.sort()
    max_val = candidate[-1]
    
    # Try adding a value v in [1, max_val + 10]
    for v in range(1, max_val + 10):
        if v in candidate:
            continue
        new_set = candidate + [v]
        new_set.sort()
        s = score_set(new_set)
        if s > best_score:
            best_score = s
            best_set = new_set
            # Restart local search with this new set
            current_set = new_set[:]
            current_score = s
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(1, len(current_set)):
                    orig_val = current_set[i]
                    for delta in range(-10, 11):
                        if delta == 0:
                            continue
                        new_val = orig_val + delta
                        if new_val < 1:
                            continue
                        if new_val in current_set:
                            continue
                        
                        new_set = current_set[:]
                        new_set[i] = new_val
                        new_set.sort()
                        
                        s = score_set(new_set)
                        if s > current_score:
                            current_score = s
                            current_set = new_set
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = current_set[:]
                            break

    return best_set
