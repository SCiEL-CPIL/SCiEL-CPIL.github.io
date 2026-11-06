"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-subsets of [16] as bitmasks
ALL_MASKS = []
N_MASKS = 0

def _init_masks():
    global ALL_MASKS, N_MASKS
    if ALL_MASKS:
        return
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        ALL_MASKS.append(mask)
    N_MASKS = len(ALL_MASKS)

def _check_conflict(new_mask, selected_masks):
    """
    Check if new_mask forms a sunflower with any pair in selected_masks.
    Sunflower condition for A, B, C: A∩B == A∩C == B∩C.
    """
    n_sel = len(selected_masks)
    for i in range(n_sel):
        a = selected_masks[i]
        ia = a & new_mask
        for j in range(i + 1, n_sel):
            b = selected_masks[j]
            ib = b & new_mask
            if ia == ib:
                if (a & b) == ia:
                    return True
    return False

def _greedy_construct(order):
    """
    Greedily construct a family using the given order of indices.
    Returns list of selected indices.
    """
    selected_indices = []
    selected_masks = []
    
    for u in order:
        mask_u = ALL_MASKS[u]
        
        # Check for conflicts with all pairs in selected
        conflict = False
        for i in range(len(selected_masks)):
            a = selected_masks[i]
            ia = a & mask_u
            for j in range(i + 1, len(selected_masks)):
                b = selected_masks[j]
                ib = b & mask_u
                if ia == ib:
                    if (a & b) == ia:
                        conflict = True
                        break
            if conflict:
                break
        
        if not conflict:
            selected_indices.append(u)
            selected_masks.append(mask_u)
            
    return selected_indices

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    _init_masks()
    
    best_size = 0
    best_set = []
    
    indices = list(range(N_MASKS))
    
    # Strategy 1: Deterministic greedy
    selected = _greedy_construct(indices)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected[:]
        
    # Strategy 2: Randomized greedy restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        random.seed(42 + restart_count)
        random.shuffle(indices)
        
        selected = _greedy_construct(indices)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
            if best_size >= 250:
                break

    # Convert indices back to sorted lists of elements
    result = []
    for idx in best_set:
        mask = ALL_MASKS[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
