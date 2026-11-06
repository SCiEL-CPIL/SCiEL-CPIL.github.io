"""Randomized greedy with bitmasks for speed and diversity."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    """Construct a large sunflower-free family using randomized greedy."""
    # Precompute all 4-subsets of [1..16] as bitmasks
    all_masks = []
    for combo in itertools.combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    n_sets = len(all_masks)
    indices = list(range(n_sets))
    
    best_size = 0
    best_family_masks = []
    
    # Run multiple randomized greedy passes
    # 100 runs should be fast enough with bitmasks
    for _ in range(100):
        random.shuffle(indices)
        chosen = []
        
        # Local function for clarity, but inlined for speed in loop below?
        # We will inline the check logic inside the loop for performance.
        
        for idx in indices:
            new_mask = all_masks[idx]
            
            # Check if new_mask can be added to chosen
            # Condition: No a, b in chosen such that (a&b) == (a&new) == (b&new)
            valid = True
            k = len(chosen)
            # Optimization: if k < 2, it's always valid
            if k >= 2:
                for i in range(k):
                    a = chosen[i]
                    an = a & new_mask
                    for j in range(i + 1, k):
                        b = chosen[j]
                        ab = a & b
                        if ab == an and ab == (b & new_mask):
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                chosen.append(new_mask)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family_masks = chosen

    # Convert best masks back to sorted lists of elements
    result = []
    for mask in best_family_masks:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
# EVOLVE-BLOCK-END
