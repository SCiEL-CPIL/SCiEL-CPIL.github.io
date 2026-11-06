"""Rewritten program for sunflower-free 4-uniform family on [1,16]."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # 1. Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    n_total = len(all_sets)  # 1820
    
    # 2. Precompute pairwise intersections
    # pair_int[i][j] = all_sets[i] & all_sets[j]
    pair_int = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        si = all_sets[i]
        for j in range(i + 1, n_total):
            inter = si & all_sets[j]
            pair_int[i][j] = inter
            pair_int[j][i] = inter
    
    best_family = []
    best_size = 0
    
    # Helper to check if a set index can be added to a family (list of indices)
    def can_add(s_idx, family_indices):
        n_fam = len(family_indices)
        if n_fam < 2:
            return True
        
        # For each A in family
        for i in range(n_fam):
            A = family_indices[i]
            I_AS = pair_int[A][s_idx]
            
            # For each B in family
            for j in range(n_fam):
                if i == j:
                    continue
                B = family_indices[j]
                # Check if A, B, S form a sunflower
                # Condition: A&B == B&S == A&S
                if pair_int[A][B] == I_AS and pair_int[B][s_idx] == I_AS:
                    return False
        return True

    def greedy_construct(order):
        family = []
        family_set = set()
        for idx in order:
            if time.time() > deadline:
                break
            if idx in family_set:
                continue
            if can_add(idx, family):
                family.append(idx)
                family_set.add(idx)
        return family

    def local_search(family, max_iters=500):
        family = family[:]
        family_set = set(family)
        n_total = len(all_sets)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 2:
            improved = False
            iters += 1
            
            # Try to add any set
            candidates = [i for i in range(n_total) if i not in family_set]
            random.shuffle(candidates)
            for idx in candidates:
                if can_add(idx, family):
                    family.append(idx)
                    family_set.add(idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add one
            # Pick a random set to remove
            if not family:
                break
            remove_idx = random.choice(family)
            temp_family = [x for x in family if x != remove_idx]
            temp_set = set(temp_family)
            
            # Find a set to add
            candidates = [i for i in range(n_total) if i not in temp_set]
            random.shuffle(candidates)
            found = False
            for idx in candidates:
                if can_add(idx, temp_family):
                    family = temp_family
                    family.append(idx)
                    family_set = set(family)
                    improved = True
                    found = True
                    break
            
            if not found and iters > 100:
                # Stuck, break
                break

        return family

    # Run multiple greedy passes
    n_attempts = 30
    for attempt in range(n_attempts):
        if time.time() > deadline - 10:
            break
        
        if attempt == 0:
            order = list(range(n_total))
        elif attempt == 1:
            order = list(range(n_total - 1, -1, -1))
        else:
            order = list(range(n_total))
            random.shuffle(order)
        
        family = greedy_construct(order)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        if best_size >= 250:
            break
        
        # Local search on the result
        if len(family) > 0 and time.time() < deadline - 5:
            improved_family = local_search(family, max_iters=200)
            if len(improved_family) > best_size:
                best_size = len(improved_family)
                best_family = improved_family[:]

    # Convert back to sorted lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
