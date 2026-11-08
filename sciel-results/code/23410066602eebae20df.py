"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate all weight-4 vectors with entries +/-1
    w4_list = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            w4_list.append(v)
    
    w4_arr = np.array(w4_list, dtype=np.int64)
    n = len(w4_arr)  # 5280
    
    # Precompute dot product matrix
    dots = w4_arr @ w4_arr.T  # (5280, 5280)
    
    # Conflict: dot > 2
    conflict = (dots > 2).astype(np.int8)  # (5280, 5280)
    
    # Greedy with minimum-degree ordering
    conflict_count = conflict.sum(axis=1)
    order = np.argsort(conflict_count, kind='stable')
    
    selected = []
    selected_mask = np.zeros(n, dtype=bool)
    
    for idx in order:
        if time.time() > deadline - 5:
            break
        if not selected:
            selected.append(idx)
            selected_mask[idx] = True
            continue
        # Check if idx conflicts with any selected (vectorized)
        sel_indices = np.where(selected_mask)[0]
        if not np.any(conflict[idx, sel_indices]):
            selected.append(idx)
            selected_mask[idx] = True
    
    best_score = len(selected)
    best_mask = selected_mask.copy()
    
    # Local search: try to improve by swaps
    unselected = np.where(~selected_mask)[0]
    
    for _round in range(3):
        if time.time() > deadline - 3:
            break
        
        found_improvement = False
        for u in unselected:
            if time.time() > deadline - 3:
                break
            
            # Conflicts of u with selected
            sel_indices = np.where(selected_mask)[0]
            conflicting = sel_indices[conflict[u, sel_indices] == 1]
            
            if len(conflicting) == 0:
                selected_mask[u] = True
                found_improvement = True
                continue
            
            # Try removing each conflicting selected vector
            for s in conflicting:
                if time.time() > deadline - 3:
                    break
                
                # Check if u conflicts with (selected - {s})
                remaining_sel = sel_indices[sel_indices != s]
                if len(remaining_sel) == 0 or not np.any(conflict[u, remaining_sel]):
                    # Valid swap: remove s, add u
                    selected_mask[s] = False
                    selected_mask[u] = True
                    found_improvement = True
                    unselected = np.where(~selected_mask)[0]
                    break
            
            if found_improvement:
                break
    
    if selected_mask.sum() > best_score:
        best_score = selected_mask.sum()
        best_mask = selected_mask.copy()
    
    return w4_arr[best_mask]

# EVOLVE-BLOCK-END
