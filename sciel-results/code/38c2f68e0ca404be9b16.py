import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 12
    
    # Phase 1: Greedy construction with efficient collision check
    current_set = []
    current_sums = {0}
    
    for k in range(n):
        candidate = 1
        while True:
            if time.time() > deadline:
                break
            
            if candidate in current_set:
                candidate += 1
                continue
            
            # Check if candidate causes collision
            # Collision exists if there exist s1, s2 in current_sums such that s1 + candidate = s2
            # i.e., candidate = s2 - s1 for some s1 != s2
            # More efficiently: check if any s + candidate is already in current_sums
            collision = False
            for s in current_sums:
                if s + candidate in current_sums:
                    collision = True
                    break
            
            if not collision:
                current_set.append(candidate)
                current_sums = current_sums | {s + candidate for s in current_sums}
                break
            else:
                candidate += 1
    
    if len(current_set) < n:
        # Fallback: use powers of 2
        return [2**i for i in range(n)]
    
    best_set = list(current_set)
    best_max = max(current_set)
    
    # Phase 2: Local search to try to improve
    # Try swapping elements or replacing with smaller values
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try each position
        for pos in range(len(current_set)):
            old_val = current_set[pos]
            
            # Try smaller values
            for new_val in range(1, old_val):
                if new_val in current_set:
                    continue
                
                # Check if replacing current_set[pos] with new_val maintains distinct subset sums
                # Remove old_val, add new_val, check all subset sums are distinct
                test_set = list(current_set)
                test_set[pos] = new_val
                
                # Compute all subset sums
                test_sums = {0}
                for v in test_set:
                    test_sums = test_sums | {s + v for s in test_sums}
                
                if len(test_sums) == 2**len(test_set):
                    # Valid replacement
                    current_set = test_set
                    best_set = list(current_set)
                    best_max = max(current_set)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_set
