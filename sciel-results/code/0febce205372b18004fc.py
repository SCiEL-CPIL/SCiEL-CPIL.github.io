import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # 1. Deterministic Greedy (Baseline)
    chosen, counts = [], {}
    for x in range(1, 1001):
        new_sums = [x + y for y in chosen] + [2 * x]
        valid = True
        for s in new_sums:
            if counts.get(s, 0) + 1 > 2:
                valid = False
                break
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
                
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_set = chosen[:]

    # 2. Randomized Greedy with Multiple Restarts
    # We run this loop until the deadline is reached.
    
    while time.time() < deadline:
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        current_set = []
        counts = {}
        
        for x in candidates:
            valid = True
            
            # Check sums with existing elements
            for y in current_set:
                s = x + y
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            
            # Check sum with itself
            if valid:
                s = 2 * x
                if counts.get(s, 0) + 1 > 2:
                    valid = False
            
            if valid:
                # Update counts for the new sums formed by adding x
                counts[2 * x] = counts.get(2 * x, 0) + 1
                for y in current_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return best_set
