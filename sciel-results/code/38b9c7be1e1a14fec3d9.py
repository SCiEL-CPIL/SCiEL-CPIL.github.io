"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    n = 16
    k = 4
    max_size = 250
    
    # Generate all 4-element subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    total = len(all_sets)
    
    # Precompute intersection sizes between all pairs
    # inter[i][j] = popcount(all_sets[i] & all_sets[j])
    inter = [[0] * total for _ in range(total)]
    for i in range(total):
        mi = all_sets[i]
        for j in range(i + 1, total):
            inter[i][j] = bin(mi & all_sets[j]).count('1')
            inter[j][i] = inter[i][j]
    
    def is_valid_triple(i, j, k_idx):
        """Check if sets i, j, k_idx form a bad triple (all three pairwise intersections equal)"""
        return inter[i][j] == inter[i][k_idx] == inter[j][k_idx]
    
    # Greedy with multiple restarts and different orderings
    best_family = []
    
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        # Try different random orderings
        order = list(range(total))
        if trial == 0:
            pass  # natural order
        elif trial == 1:
            random.shuffle(order)
        else:
            # Mix: some random, some structured
            random.shuffle(order)
        
        chosen_indices = []
        chosen_masks = []
        
        for idx in order:
            if len(chosen_indices) >= max_size:
                break
            if time.time() > deadline - 2:
                break
            
            # Check if adding idx creates a bad triple with any pair in chosen
            valid = True
            mask = all_sets[idx]
            
            for a_pos in range(len(chosen_indices)):
                ia = chosen_indices[a_pos]
                for b_pos in range(a_pos + 1, len(chosen_indices)):
                    ib = chosen_indices[b_pos]
                    if is_valid_triple(ia, ib, idx):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                chosen_indices.append(idx)
                chosen_masks.append(mask)
        
        if len(chosen_indices) > len(best_family):
            best_family = list(chosen_indices)
            if len(best_family) >= max_size:
                break
    
    # Convert back to sorted lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for x in range(n):
            if mask & (1 << x):
                elems.append(x + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
