import random
import time
from itertools import combinations

def construct():
    # Generate all 4-subsets of [1, 16] as bitmasks
    n = 16
    k = 4
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_sets.append(mask)
        
    # Shuffle for randomization
    random.shuffle(all_sets)
    
    # Helper to check if a set can be added to a family
    def is_valid_add(s, fam):
        m = len(fam)
        for i in range(m):
            a = fam[i]
            inter_as = a & s
            for j in range(i + 1, m):
                b = fam[j]
                inter_bs = b & s
                if inter_as == inter_bs:
                    if (a & b) == inter_as:
                        return False
        return True

    # Helper to check if a swap (remove a, add b) is valid
    def is_valid_swap(a, b, fam, idx_a):
        m = len(fam)
        for i in range(m):
            if i == idx_a:
                continue
            ai = fam[i]
            inter_ai_b = ai & b
            for j in range(i + 1, m):
                if j == idx_a:
                    continue
                aj = fam[j]
                inter_aj_b = aj & b
                if inter_ai_b == inter_aj_b:
                    if (ai & aj) == inter_ai_b:
                        return False
        return True

    # Greedy Initialization
    family = []
    family_set = set()
    
    for s in all_sets:
        if is_valid_add(s, family):
            family.append(s)
            family_set.add(s)
            
    best_family = list(family)
    best_size = len(family)
    
    start_time = time.time()
    time_limit = 50.0 # Safety margin
    
    # Local Search: Swap moves
    # We try to swap a set in family with a set not in family
    while time.time() - start_time < time_limit:
        if len(family) < 2:
            break
            
        # Pick random A from family
        idx_a = random.randint(0, len(family) - 1)
        a = family[idx_a]
        
        # Pick random B from all_sets not in family
        b = None
        for _ in range(10):
            cand = random.choice(all_sets)
            if cand not in family_set:
                b = cand
                break
        if b is None:
            continue
            
        # Check if swap is valid
        if is_valid_swap(a, b, family, idx_a):
            # Perform swap
            family[idx_a] = b
            family_set.remove(a)
            family_set.add(b)
            
            if len(family) > best_size:
                best_size = len(family)
                best_family = list(family)
                
    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
