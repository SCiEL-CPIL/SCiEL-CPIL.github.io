import time
import random
from itertools import combinations

# Pre-compute all 4-subsets of [1, 16] as bitmasks
ALL_SETS = []
for comb in combinations(range(16), 4):
    mask = 0
    for i in comb:
        mask |= (1 << i)
    ALL_SETS.append(mask)

def is_sunflower(a, b, c):
    return (a & b) == (a & c) == (b & c)

def mask_to_list(mask):
    return [i + 1 for i in range(16) if mask & (1 << i)]

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s limit
    best_family = []
    max_size = 250
    
    # If we find the maximum possible, we can stop
    # But we don't know if 250 is the true max, just the cap.
    # The problem says "at most 250 members".
    # So 250 is the global upper bound for the value.
    
    for _ in range(10000):
        if time.time() - start_time > deadline:
            break
        
        random.shuffle(ALL_SETS)
        current_family = []
        
        for s in ALL_SETS:
            if len(current_family) >= max_size:
                break
            
            # Check if s can be added
            valid = True
            n = len(current_family)
            # Iterate over pairs in current_family
            for i in range(n):
                a = current_family[i]
                for j in range(i):
                    b = current_family[j]
                    if is_sunflower(a, b, s):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_family.append(s)
        
        if len(current_family) > len(best_family):
            best_family = current_family
            if len(best_family) >= max_size:
                break

    # Convert best_family masks back to sorted lists
    return [mask_to_list(m) for m in best_family]
