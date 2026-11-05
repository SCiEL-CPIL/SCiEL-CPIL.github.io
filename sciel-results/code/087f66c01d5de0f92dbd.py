import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1, 16] as bitmasks
ALL_SETS = []
for combo in combinations(range(16), 4):
    mask = 0
    for x in combo:
        mask |= (1 << x)
    ALL_SETS.append(mask)

def is_valid_addition(family, new_set):
    n = len(family)
    for i in range(n):
        A = family[i]
        A_new = A & new_set
        for j in range(i + 1, n):
            B = family[j]
            # Check if B & new_set == A_new AND A & B == A_new
            if (B & new_set) == A_new:
                if (A & B) == A_new:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_family = []
    best_size = 0
    
    # Run randomized greedy multiple times
    # We don't know how many we can fit, so we loop until time runs out.
    
    while time.time() < deadline:
        current_sets = ALL_SETS[:]
        random.shuffle(current_sets)
        
        family = []
        for s in current_sets:
            if time.time() > deadline - 2.0:
                break
            
            if is_valid_addition(family, s):
                family.append(s)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
            # If we found a very large family, we might want to spend more time on it?
            # No, just keep restarting to see if we can beat it.
            
    # Convert best_family masks back to sorted lists of 1..16
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
