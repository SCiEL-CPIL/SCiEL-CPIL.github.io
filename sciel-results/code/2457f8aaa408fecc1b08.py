import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
        
    total = len(all_masks)
    
    # Precompute intersections
    # inter[i][j] is the bitmask of intersection of all_masks[i] and all_masks[j]
    inter = [[0] * total for _ in range(total)]
    for i in range(total):
        mi = all_masks[i]
        inter[i][i] = mi
        for j in range(i + 1, total):
            val = mi & all_masks[j]
            inter[i][j] = val
            inter[j][i] = val
            
    def is_compatible(cand_idx, family_indices):
        # Bucket family members by their intersection with cand_idx
        groups = {}
        for f_idx in family_indices:
            K = inter[f_idx][cand_idx]
            if K not in groups:
                groups[K] = []
            groups[K].append(f_idx)
            
        for K, group in groups.items():
            L = len(group)
            if L < 2:
                continue
            # Check all pairs in group
            for i in range(L):
                ai = group[i]
                for j in range(i + 1, L):
                    bj = group[j]
                    if inter[ai][bj] == K:
                        return False
        return True

    def run_greedy(order_indices):
        family = []
        for idx in order_indices:
            if time.time() > deadline - 2:
                break
            if is_compatible(idx, family):
                family.append(idx)
        return family

    best_family = []
    
    # 1. Lexicographic Order
    order = list(range(total))
    if time.time() < deadline - 5:
        fam = run_greedy(order)
        if len(fam) > len(best_family):
            best_family = fam[:]

    # 2. Reverse Lexicographic Order
    if time.time() < deadline - 5:
        order_rev = list(reversed(range(total)))
        fam = run_greedy(order_rev)
        if len(fam) > len(best_family):
            best_family = fam[:]

    # 3. Random Permutations
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        perm = list(range(n))
        rng.shuffle(perm)
        
        # Create sorting keys
        keys = []
        for idx in range(total):
            mask = all_masks[idx]
            elems = []
            for i in range(n):
                if mask & (1 << i):
                    elems.append(perm[i])
            elems.sort()
            keys.append((tuple(elems), idx))
            
        keys.sort()
        order_perm = [idx for _, idx in keys]
        
        fam = run_greedy(order_perm)
        if len(fam) > len(best_family):
            best_family = fam[:]
            
    # Convert best_family to list of sorted 1-indexed lists
    result = []
    for mask in best_family:
        elements = [i+1 for i in range(n) if (mask & (1 << i))]
        result.append(elements)
        
    return result
