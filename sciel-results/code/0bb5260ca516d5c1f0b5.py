import time
import random

def count_aps_optimized(s_list):
    """
    Calculate the number of distinct differences d > 0 such that
    a, a+d, a+2d are all in the set s_list.
    Uses a boolean array for O(1) membership checks.
    """
    max_val = 512
    present = [False] * (max_val + 1)
    for x in s_list:
        present[x] = True
    
    diffs = set()
    n = len(s_list)
    # To ensure we find all d, we iterate through all pairs (a, b) with a < b
    # and check if c = a + 2*(b-a) is in the set.
    # This covers all 3-APs.
    for i in range(n):
        a = s_list[i]
        for j in range(n):
            if i == j:
                continue
            b = s_list[j]
            if b <= a:
                continue
            d = b - a
            c = a + 2 * d
            if c <= max_val and present[c]:
                diffs.add(d)
    return len(diffs)

def construct():
    N = 32
    MAX_VAL = 512
    start_time = time.time()
    time_limit = 115.0
    
    # Initialize with a dense set which has many APs
    # {1, 2, ..., 32} has 15 distinct differences
    current_set = list(range(1, N + 1))
    current_score = count_aps_optimized(current_set)
    
    best_set = list(current_set)
    best_score = current_score
    
    # Hill Climbing
    # We will try all single-element swaps
    # To speed up, we pre-calculate the list of all possible values
    all_values = list(range(1, MAX_VAL + 1))
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        best_delta = 0
        best_swap = None # (old_val, new_val)
        
        # Iterate over each element to remove
        for i in range(N):
            old_val = current_set[i]
            
            # Iterate over each candidate to add
            # Optimization: We can skip values that are "too far" or similar, 
            # but for correctness we check all.
            for new_val in all_values:
                if new_val == old_val:
                    continue
                # Check if new_val is already in the set
                # Since N is small, we can check if new_val is in current_set
                # But we are removing old_val, so if new_val is in current_set and new_val != old_val, it's invalid
                if new_val in current_set:
                    continue
                
                # Create new set
                # Instead of creating a new list, we can simulate
                # But count_aps_optimized takes a list.
                # We can create a modified list
                new_set = current_set.copy()
                new_set[i] = new_val
                
                new_score = count_aps_optimized(new_set)
                
                delta = new_score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (old_val, new_val)
        
        if best_swap is not None:
            old_val, new_val = best_swap
            current_set.remove(old_val)
            current_set.append(new_val)
            current_score += best_delta
            improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                
    return best_set
