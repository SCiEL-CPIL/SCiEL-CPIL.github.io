import numpy as np
import time
from itertools import combinations
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin before 630s kill
    
    # Generate all 4-subsets of {0, ..., 10}
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)
    
    # Precompute conflict bitsets: conflict[i] has bit j set if sets i and j share >= 3 indices
    conflict = [0] * n_sets
    for i in range(n_sets):
        si = all_4sets[i]
        mask = 0
        for j in range(n_sets):
            if i == j:
                continue
            sj = all_4sets[j]
            # Count common elements
            common = 0
            for a in si:
                if a in sj:
                    common += 1
                    if common >= 3:
                        break
            if common >= 3:
                mask |= (1 << j)
        conflict[i] = mask
    
    # Bitset-based greedy
    def greedy_bitset(order, rng):
        selected = []
        available = (1 << n_sets) - 1
        for idx in order:
            if available & (1 << idx):
                selected.append(idx)
                available &= ~conflict[idx]
                available &= ~(1 << idx)
        return selected
    
    # Try to add a valid set to selection
    def try_add(selection, sel_set, rng):
        for j in range(n_sets):
            if j in sel_set:
                continue
            valid = True
            for k in selection:
                if conflict[k] & (1 << j):
                    valid = False
                    break
            if valid:
                selection.append(j)
                sel_set.add(j)
                return True
        return False
    
    # 1-swap: remove one, add another
    def one_swap(selection, sel_set, rng):
        if not selection:
            return False
        i_idx = rng.randint(0, len(selection) - 1)
        i_val = selection[i_idx]
        
        # Remove i_val
        selection.pop(i_idx)
        sel_set.remove(i_val)
        
        # Find valid j to add
        forbidden = 0
        for k in selection:
            forbidden |= conflict[k]
        
        candidates = []
        for j in range(n_sets):
            if j in sel_set:
                continue
            if forbidden & (1 << j):
                continue
            candidates.append(j)
        
        if candidates:
            j_val = rng.choice(candidates)
            selection.append(j_val)
            sel_set.add(j_val)
            return True
        else:
            # Revert
            selection.append(i_val)
            sel_set.add(i_val)
            return False
    
    # Find best selection using multiple random greedy + local search
    rng = random.Random(42)
    best_selection = []
    best_size = 0
    
    # Multiple random greedy initializations
    for trial in range(200):
        if time.time() > deadline - 30:
            break
        order = list(range(n_sets))
        rng.shuffle(order)
        sel = greedy_bitset(order, rng)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selection = sel[:]
    
    # Local search on best selection
    selection = best_selection[:]
    sel_set = set(selection)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Try to add
        if try_add(selection, sel_set, rng):
            improved = True
            continue
        
        # Try 1-swap
        if one_swap(selection, sel_set, rng):
            improved = True
            continue
        
        # Try random 1-swap (different random choice)
        if selection:
            i_idx = rng.randint(0, len(selection) - 1)
            i_val = selection[i_idx]
            selection.pop(i_idx)
            sel_set.remove(i_val)
            
            forbidden = 0
            for k in selection:
                forbidden |= conflict[k]
            
            candidates = []
            for j in range(n_sets):
                if j in sel_set:
                    continue
                if forbidden & (1 << j):
                    continue
                candidates.append(j)
            
            if candidates:
                j_val = rng.choice(candidates)
                selection.append(j_val)
                sel_set.add(j_val)
                improved = True
            else:
                selection.append(i_val)
                sel_set.add(i_val)
    
    # Generate points from best_selection
    points = []
    for idx in best_selection:
        i, j, k, l = all_4sets[idx]
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
    
    # Fallback: if no points found, use 2-support vectors
    if not points:
        for i in range(d):
            for j in range(i + 1, d):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
