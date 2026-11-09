import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    def can_add(family, new_set):
        # Group family members by their intersection with new_set
        groups = {}
        for a in family:
            inter = a & new_set
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(a)
        
        # For each group with intersection K, check if any pair A, B in group has A & B == K
        for K, members in groups.items():
            L = len(members)
            for i in range(L):
                a = members[i]
                for j in range(i + 1, L):
                    b = members[j]
                    if (a & b) == K:
                        return False
        return True

    best_family = []
    
    while time.time() < deadline:
        shuffled_sets = all_sets[:]
        random.shuffle(shuffled_sets)
        
        current_family = []
        
        for s in shuffled_sets:
            if time.time() > deadline:
                break
            
            if can_add(current_family, s):
                current_family.append(s)
        
        if len(current_family) > len(best_family):
            best_family = current_family[:]
            
    result = []
    for mask in best_family:
        subset = []
        for i in range(16):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
