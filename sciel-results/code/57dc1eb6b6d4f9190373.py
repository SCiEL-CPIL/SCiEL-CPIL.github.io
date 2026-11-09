"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import random

def _greedy_build_bitmasks(all_masks, order):
    """Greedy: iterate in given order, add set if no sunflower with any existing pair."""
    chosen_masks = []
    n_chosen = 0
    
    for idx in order:
        s = all_masks[idx]
        conflict_found = False
        
        # Check against all pairs in chosen
        for i in range(n_chosen):
            if conflict_found:
                break
            a = chosen_masks[i]
            as_ = a & s
            for j in range(i + 1, n_chosen):
                b = chosen_masks[j]
                ab = a & b
                bs = b & s
                if ab == as_ == bs:
                    conflict_found = True
                    break
        
        if not conflict_found:
            chosen_masks.append(s)
            n_chosen += 1
    
    return chosen_masks

def construct():
    # Generate all 4-subsets of [1, 16] as bitmasks
    # 16 elements fit in 16 bits.
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for bit in c:
            mask |= (1 << bit)
        all_masks.append(mask)
    
    n_sets = len(all_masks)
    
    best_size = 0
    best_chosen_masks = []
    
    rng = random.Random(42)
    
    # Increase trials due to faster bitmask operations
    # 2184 sets. 
    # Greedy is O(N * K^2). 
    # With bitmasks, it's very fast.
    # Try 1000 restarts.
    
    for trial in range(1000):
        order = list(range(n_sets))
        rng.shuffle(order)
        
        chosen_masks = _greedy_build_bitmasks(all_masks, order)
        
        if len(chosen_masks) > best_size:
            best_size = len(chosen_masks)
            best_chosen_masks = chosen_masks[:]
            
        # Early stop if we hit the theoretical max? 
        # Max is 250. If we reach 250, we can stop.
        if best_size >= 250:
            break
    
    # Convert bitmasks back to sorted lists of 1-based indices
    result = []
    for mask in best_chosen_masks:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(elements)
        
    return result
# EVOLVE-BLOCK-END
