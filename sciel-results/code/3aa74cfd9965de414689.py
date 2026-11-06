import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Increased budget to allow more iterations
    
    # Initial set: Interval 1..32
    current_set = set(range(1, 33))
    
    # Initialize counts
    max_diff = 512
    count = [0] * (max_diff + 1)
    current_score = 0
    
    # Helper to update counts for adding/removing an element
    # Returns the change in score
    def update_counts(val, S, adding):
        delta_score = 0
        sign = 1 if adding else -1
        # d can range from 1 to max_diff
        # We only need to check d such that the AP stays within 1..512?
        # No, S is a subset of 1..512, so if a, a+d, a+2d are in S, they are in 1..512.
        # But d can be up to 511.
        
        for d in range(1, max_diff + 1):
            # Check if val is part of an AP with difference d in S
            # 1. val is a
            # 2. val is a+d
            # 3. val is a+2d
            
            # Note: We check membership in S. 
            # If adding, S does not contain val yet.
            # If removing, S contains val.
            
            # Case 1: val is start
            # Need val+d in S and val+2d in S
            if val + d <= 512 and val + 2*d <= 512:
                if (val + d in S) and (val + 2*d in S):
                    # If adding, this AP will exist.
                    # If removing, this AP will cease to exist (provided the other elements are still in S, which they are).
                    count[d] += sign
                    if sign == 1:
                        if count[d] == 1: delta_score += 1
                    else:
                        if count[d] == 0: delta_score -= 1
                        
            # Case 2: val is middle
            # Need val-d in S and val+d in S
            if val - d >= 1 and val + d <= 512:
                if (val - d in S) and (val + d in S):
                    count[d] += sign
                    if sign == 1:
                        if count[d] == 1: delta_score += 1
                    else:
                        if count[d] == 0: delta_score -= 1
                        
            # Case 3: val is end
            # Need val-2d in S and val-d in S
            if val - 2*d >= 1 and val - d >= 1:
                if (val - 2*d in S) and (val - d in S):
                    count[d] += sign
                    if sign == 1:
                        if count[d] == 1: delta_score += 1
                    else:
                        if count[d] == 0: delta_score -= 1
                        
        return delta_score

    # Initialize counts for the starting set
    for x in current_set:
        current_score += update_counts(x, current_set, True)
        
    best_set = set(current_set)
    best_score = current_score
    
    # Local Search
    while time.time() < deadline:
        # Pick a random element to replace
        elem_to_replace = random.choice(list(current_set))
        
        # Pick a random new value
        new_val = random.randint(1, 512)
        if new_val in current_set:
            continue
            
        # Try replacement
        # 1. Remove elem_to_replace
        delta_remove = update_counts(elem_to_replace, current_set, False)
        current_set.remove(elem_to_replace)
        current_score += delta_remove
        
        # 2. Add new_val
        delta_add = update_counts(new_val, current_set, True)
        current_set.add(new_val)
        current_score += delta_add
        
        if current_score >= best_score:
            best_score = current_score
            best_set = set(current_set)
        else:
            # Revert
            # Remove new_val
            delta_revert_add = update_counts(new_val, current_set, False)
            current_set.remove(new_val)
            current_score += delta_revert_add
            
            # Add elem_to_replace
            delta_revert_remove = update_counts(elem_to_replace, current_set, True)
            current_set.add(elem_to_replace)
            current_score += delta_revert_remove

    return sorted(best_set)
