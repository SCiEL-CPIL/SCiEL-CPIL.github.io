import time
import random
import numpy as np

N = 512
DEADLINE_MARGIN = 5

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_MARGIN

    def elapsed_ok():
        return (time.time() - t0) < deadline

    # Pre-compute all 4-APs in [1, N]
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # Store as list of tuples of 4 integers
    all_aps = []
    ap_index = {} # map number -> list of ap indices
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_aps)
            all_aps.append(ap)
            for x in ap:
                if x not in ap_index:
                    ap_index[x] = []
                ap_index[x].append(idx)
            
    num_aps = len(all_aps)
    
    # Helper to check if adding x is valid given current in_set
    def can_add(x, in_set):
        if x not in ap_index:
            return True
        for idx in ap_index[x]:
            ap = all_aps[idx]
            # Check if the other 3 elements are in set
            count = 0
            for y in ap:
                if y != x:
                    if in_set[y]:
                        count += 1
                    else:
                        break
            if count == 3:
                return False
        return True

    # Initial set: Base-4 digits 0 and 1
    in_set = np.zeros(N + 1, dtype=bool)
    for i in range(1, N + 1):
        # Check if base 4 representation has only 0s and 1s
        x = i
        valid = True
        while x > 0:
            digit = x % 4
            if digit > 1:
                valid = False
                break
            x //= 4
        if valid:
            in_set[i] = True
            
    # Greedy pass 1: Add any number that doesn't conflict
    # Order: increasing
    for x in range(1, N + 1):
        if not elapsed_ok():
            break
        if not in_set[x] and can_add(x, in_set):
            in_set[x] = True

    best_set = {x for x in range(1, N + 1) if in_set[x]}
    best_size = len(best_set)
    
    # Local Search: Try to add more
    # We can try random orders or specific heuristics
    # Let's try a few random restarts of the greedy part on the remaining numbers
    # Or just continue adding if possible
    
    # A better local search: 
    # Try to find a pair (remove y, add x) that increases size? No, size is same.
    # Try to find a triple (remove y, add x, z)?
    
    # Given the small N, we can just try to add all possible remaining numbers in random order multiple times.
    
    rng = random.Random(42)
    
    for trial in range(50):
        if not elapsed_ok():
            break
        
        # Reset to base-4 set
        in_set = np.zeros(N + 1, dtype=bool)
        for i in range(1, N + 1):
            x = i
            valid = True
            while x > 0:
                digit = x % 4
                if digit > 1:
                    valid = False
                    break
                x //= 4
            if valid:
                in_set[i] = True
        
        # Get list of candidates
        candidates = [x for x in range(1, N + 1) if not in_set[x]]
        rng.shuffle(candidates)
        
        # Greedy add
        for x in candidates:
            if not elapsed_ok():
                break
            if can_add(x, in_set):
                in_set[x] = True
        
        current_set = {x for x in range(1, N + 1) if in_set[x]}
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    return sorted(best_set)
