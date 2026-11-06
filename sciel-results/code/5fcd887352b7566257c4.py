"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
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
    
    # Precompute pairwise sunflower conflicts
    # Three sets form a sunflower if all three pairwise intersections are equal
    # For 4-element sets, this means: A∩B = A∩C = B∩C
    # We'll build a conflict graph: edge between i,j if there exists k such that {i,j,k} is a sunflower
    # But that's O(n^3). Instead, for greedy we check incrementally.
    
    def is_sunflower(mask_a, mask_b, mask_c):
        ab = mask_a & mask_b
        ac = mask_a & mask_c
        bc = mask_b & mask_c
        return ab == ac == bc
    
    def can_add(mask_new, chosen_masks):
        for i in range(len(chosen_masks)):
            for j in range(i+1, len(chosen_masks)):
                if is_sunflower(chosen_masks[i], chosen_masks[j], mask_new):
                    return False
        return True
    
    # Greedy with randomization and multiple restarts
    best_family = []
    best_size = 0
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(42 + restart)
        shuffled = all_subsets[:]
        random.shuffle(shuffled)
        
        chosen = []
        chosen_set = set()
        
        for mask in shuffled:
            if time.time() > deadline:
                break
            if mask in chosen_set:
                continue
            if can_add(mask, chosen):
                chosen.append(mask)
                chosen_set.add(mask)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
        
        # Early stopping if we've hit the max
        if best_size >= 250:
            break
    
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
