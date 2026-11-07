import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin
    
    best_set = [11, 13, 17]
    best_size = len(best_set)
    
    # Helper to get all non-empty subset sums
    def get_sums(S):
        sums = set()
        n = len(S)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sums.add(s)
        return sums

    # Check if we can add x to S with current sums
    def can_add(S, x, current_sums):
        # Generate new sums involving x
        new_sums = set()
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(x + s)
            
        # Check for overlap (duplicate sums)
        if current_sums & new_sums:
            return False, None
            
        # Check divisibility
        # We only need to check pairs where at least one sum is in new_sums
        # 1. new_sums vs current_sums
        for s_new in new_sums:
            for s_old in current_sums:
                if s_new > s_old:
                    if s_new % s_old == 0:
                        return False, None
                else:
                    if s_old % s_new == 0:
                        return False, None
                        
        # 2. new_sums vs new_sums
        new_sums_list = list(new_sums)
        for i in range(len(new_sums_list)):
            for j in range(i + 1, len(new_sums_list)):
                a, b = new_sums_list[i], new_sums_list[j]
                if a > b:
                    if a % b == 0:
                        return False, None
                else:
                    if b % a == 0:
                        return False, None
                            
        return True, new_sums

    # Randomized greedy search
    candidates = list(range(1, 257))
    
    # Try a few random restarts
    max_restarts = 1000
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        current_set = []
        current_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            valid, new_sums = can_add(current_set, x, current_sums)
            if valid:
                current_set.append(x)
                current_sums = current_sums.union(new_sums)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                    
        # Optional: Try to extend best_set found in previous restarts?
        # For now, simple restarts.

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
