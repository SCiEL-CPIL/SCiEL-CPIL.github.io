import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = []
    best_len = 0
    
    candidates = list(range(1, 257))
    
    def is_valid_addition(current_sums, new_elem):
        # current_sums is a set of all non-empty subset sums of the current set
        # new_elem is the integer to add
        
        # Generate new sums
        new_sums = {new_elem}
        for s in current_sums:
            new_sums.add(s + new_elem)
            
        # Check 1: Uniqueness (Intersection with old sums must be empty)
        if new_sums & current_sums:
            return None
            
        # Check 2: Divisibility
        # Optimization: Only check divisibility if the larger number is at least double the smaller one.
        # If b < 2a and a != b, then b % a != 0.
        
        new_list = list(new_sums)
        
        # Internal conflicts in new_sums
        for i in range(len(new_list)):
            a = new_list[i]
            for j in range(i + 1, len(new_list)):
                b = new_list[j]
                if b >= 2 * a:
                    if b % a == 0:
                        return None
                elif a >= 2 * b:
                    if a % b == 0:
                        return None
        
        # Cross conflicts between new_sums and current_sums
        for a in new_list:
            for b in current_sums:
                if b >= 2 * a:
                    if b % a == 0:
                        return None
                elif a >= 2 * b:
                    if a % b == 0:
                        return None
        
        return new_sums

    # Randomized Greedy Search with Restarts
    while time.time() < deadline:
        random.shuffle(candidates)
        current_set = []
        current_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            new_sums = is_valid_addition(current_sums, x)
            if new_sums is not None:
                current_set.append(x)
                current_sums = new_sums
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
        
        # Optional: If we found a good set, we can try to "polish" it
        # But simple restarts are often sufficient for this size.
        
    # Fallback if no set was found (should not happen, [1] is always valid)
    if not best_set:
        best_set = [1]
        
    return best_set
