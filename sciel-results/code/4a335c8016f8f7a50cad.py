import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_sets.append(mask)
    
    best_family = []
    start_time = time.time()
    time_limit = 85.0
    
    while time.time() - start_time < time_limit:
        current_family = []
        # Create a shuffled copy of all sets
        shuffled_sets = all_sets[:]
        random.shuffle(shuffled_sets)
        
        for s in shuffled_sets:
            can_add = True
            # Check against all pairs in current_family
            # Optimization: If current_family is empty or size 1, can_add is always True
            if len(current_family) >= 2:
                # Iterate over pairs
                # We can optimize by breaking early
                m = len(current_family)
                for i in range(m):
                    a = current_family[i]
                    inter_as = a & s
                    for j in range(i + 1, m):
                        b = current_family[j]
                        inter_bs = b & s
                        if inter_as == inter_bs:
                            if (a & b) == inter_as:
                                can_add = False
                                break
                    if not can_add:
                        break
            
            if can_add:
                current_family.append(s)
        
        if len(current_family) > len(best_family):
            best_family = current_family

    # Convert bitmasks to lists of 1-based integers
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
