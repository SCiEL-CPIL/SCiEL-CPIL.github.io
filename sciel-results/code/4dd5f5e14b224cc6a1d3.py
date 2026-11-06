"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 16
    k = 4
    
    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)
    
    num_subsets = len(all_subsets)
    
    def can_add(mask_new, chosen_masks):
        # Group chosen sets by their intersection with mask_new
        # For a sunflower with M, A, B: A∩M = B∩M = A∩B = I
        # So we group by I = A∩M, and within each group check if any pair has intersection I
        groups = defaultdict(list)
        for m in chosen_masks:
            inter = m & mask_new
            groups[inter].append(m)
        
        for inter, group in groups.items():
            if len(group) < 2:
                continue
            L = len(group)
            for i in range(L):
                gi = group[i]
                for j in range(i + 1, L):
                    gj = group[j]
                    if (gi & gj) == inter:
                        return False
        return True

    def try_local_improve(family):
        """Try to improve family by removing one set and adding another."""
        family_set = set(family)
        family_list = list(family)
        improved = False
        
        # Try each removal
        for i in range(len(family_list)):
            if time.time() > deadline - 5.0:
                return family_list, improved
            removed = family_list[i]
            remaining = family_list[:i] + family_list[i+1:]
            
            # Try to add a set not in family
            for mask in all_subsets:
                if time.time() > deadline - 5.0:
                    return family_list, improved
                if mask in family_set:
                    continue
                # Check if mask can be added to remaining
                if can_add(mask, remaining):
                    new_family = remaining + [mask]
                    return new_family, True
        
        return family_list, improved

    best_family = []
    best_size = 0
    
    # Strategy 1: Deterministic Greedy (Lexicographic)
    chosen = []
    for mask in all_subsets:
        if time.time() > deadline:
            break
        if can_add(mask, chosen):
            chosen.append(mask)
            
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_family = chosen[:]

    # Strategy 2: Randomized Greedy Restarts
    num_restarts = 300
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:  # 5s safety margin
            break
        
        random.seed(12345 + restart)
        shuffled = all_subsets[:]
        random.shuffle(shuffled)
        
        chosen = []
        
        for mask in shuffled:
            if time.time() > deadline - 5.0:
                break
            if can_add(mask, chosen):
                chosen.append(mask)
                
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
            if best_size >= 250:
                break

    # Strategy 3: Local improvement on best family
    if time.time() < deadline - 10.0 and best_size < 250:
        improved_family, improved = try_local_improve(best_family)
        if improved:
            best_family = improved_family
            best_size = len(best_family)
            # Try another round of improvement
            if time.time() < deadline - 5.0:
                improved_family2, improved2 = try_local_improve(best_family)
                if improved2:
                    best_family = improved_family2
                    best_size = len(best_family)

    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
