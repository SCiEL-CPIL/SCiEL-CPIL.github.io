"""Combined program: structural SAT-style greedy + refinement via numpy vectorized checks."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _make_bitmasks():
    """Generate all 4-element subsets of [1,16] as bitmasks, indexed."""
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    return np.array(all_sets, dtype=np.int64)

def _check_conflict_vectorized(c_mask, chosen_arr):
    """Check if c_mask conflicts with any pair in chosen_arr (numpy array of int64).
    Returns True if a sunflower triple exists."""
    n = len(chosen_arr)
    if n < 2:
        return False
    
    # I_i = chosen[i] & c_mask for all i
    I = chosen_arr & c_mask  # shape (n,)
    
    # For each i, check if there exists j != i such that:
    #   chosen[i] & chosen[j] == I_i  AND  chosen[j] & c_mask == I_i
    # The second condition is I_j == I_i
    # The first condition: chosen[i] & chosen[j] == I_i
    
    for i in range(n):
        if i >= n:
            break
        inter_i_c = I[i]
        # Find all j where I_j == I_i
        same_inter = (I == inter_i_c)
        # Exclude i itself
        same_inter[i] = False
        
        if not np.any(same_inter):
            continue
        
        # For those j, check if chosen[i] & chosen[j] == inter_i_c
        # Vectorized: compute chosen[i] & chosen[j] for all j where same_inter is True
        # But we can just check: for j in indices where same_inter, is chosen[i] & chosen[j] == inter_i_c?
        # Let's compute for all j: inter_i_j = chosen[i] & chosen_arr
        inter_i_j = chosen_arr[i] & chosen_arr  # shape (n,)
        
        # Conflict if there exists j != i with same_inter[j] AND inter_i_j[j] == inter_i_c
        conflict_mask = same_inter & (inter_i_j == inter_i_c)
        if np.any(conflict_mask):
            return True
    
    return False

def _greedy_build(all_masks, order, deadline):
    """Greedy: add masks in given order if no conflict. Uses numpy for speed."""
    chosen = []
    chosen_arr = np.array([], dtype=np.int64)
    
    for idx in order:
        if time.time() > deadline:
            break
        c_mask = all_masks[idx]
        
        if len(chosen) < 2:
            chosen.append(idx)
            chosen_arr = np.append(chosen_arr, c_mask)
            continue
        
        if not _check_conflict_vectorized(c_mask, chosen_arr):
            chosen.append(idx)
            chosen_arr = np.append(chosen_arr, c_mask)
    
    return chosen, chosen_arr

def _local_search(chosen_indices, chosen_arr, all_masks, n_total, deadline):
    """Try to improve by adding unchosen masks that don't conflict."""
    chosen_set = set(chosen_indices)
    
    # Build list of unchosen indices
    unchosen = [i for i in range(n_total) if i not in chosen_set]
    random.shuffle(unchosen)
    
    improved = False
    for idx in unchosen:
        if time.time() > deadline:
            break
        c_mask = all_masks[idx]
        if not _check_conflict_vectorized(c_mask, chosen_arr):
            chosen_indices.append(idx)
            chosen_arr = np.append(chosen_arr, c_mask)
            chosen_set.add(idx)
            improved = True
    
    return chosen_indices, chosen_arr, improved

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    all_masks = _make_bitmasks()
    n_total = len(all_masks)
    random.seed(42)
    
    best_indices = []
    best_arr = np.array([], dtype=np.int64)
    
    # Multiple greedy passes with different orderings
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        order = list(range(n_total))
        random.shuffle(order)
        
        chosen_indices, chosen_arr = _greedy_build(all_masks, order, deadline - 5)
        
        if len(chosen_indices) > len(best_indices):
            best_indices = chosen_indices[:]
            best_arr = chosen_arr.copy()
        
        # Local search improvement
        if time.time() > deadline - 10:
            break
        improved_indices, improved_arr, _ = _local_search(
            best_indices[:], best_arr.copy(), all_masks, n_total, deadline - 5
        )
        if len(improved_indices) > len(best_indices):
            best_indices = improved_indices
            best_arr = improved_arr
    
    # Convert bitmasks back to sorted lists
    result = []
    for idx in best_indices:
        mask = all_masks[idx]
        elems = [i+1 for i in range(16) if mask & (1 << i)]
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
