"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def _check_validity(vectors):
        """Check if the set of vectors satisfies the constraint."""
        if len(vectors) == 0:
            return True
        norms_sq = np.sum(vectors * vectors, axis=1)
        max_norm_sq = np.max(norms_sq)
        n = len(vectors)
        for i in range(n):
            for j in range(i+1, n):
                diff = vectors[i] - vectors[j]
                dist_sq = np.sum(diff * diff)
                if dist_sq < max_norm_sq - 1e-9:
                    return False
        return True
    
    # Generate candidate vectors with various sparsities
    candidates = []
    
    # Norm 1: single ±1
    for i in range(D):
        for val in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = val
            candidates.append(v)
    
    # Norm sqrt(2): two ±1
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    candidates.append(v)
    
    # Norm sqrt(3): three ±1
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i], v[j], v[k] = si, sj, sk
                            candidates.append(v)
    
    # Norm 2: one ±2
    for i in range(D):
        for val in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = val
            candidates.append(v)
    
    # Norm sqrt(5): two ±1 and one ±2? No, that's norm sqrt(1+1+4)=sqrt(6)
    # Norm sqrt(6): one ±2 and two ±1
    for i in range(D):
        for j in range(D):
            if i == j: continue
            for k in range(j+1, D):
                if k == i: continue
                for s2 in (2, -2):
                    for si in (1, -1):
                        for sj in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = s2
                            v[j] = si
                            v[k] = sj
                            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    candidates = np.unique(candidates, axis=0)
    
    norms_sq = np.sum(candidates * candidates, axis=1)
    n_candidates = len(candidates)
    
    def greedy_select(order, time_limit):
        """Greedy selection given an ordering of candidates."""
        selected = []
        max_norm_sq = 0
        
        for idx in order:
            if time.time() > time_limit:
                break
            
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            new_max_norm_sq = max(max_norm_sq, v_norm_sq)
            
            if len(selected) == 0:
                selected.append(v)
                max_norm_sq = new_max_norm_sq
                continue
            
            selected_arr = np.array(selected)
            diffs = v - selected_arr
            dists_sq = np.sum(diffs * diffs, axis=1)
            
            if np.all(dists_sq >= new_max_norm_sq - 1e-9):
                selected.append(v)
                max_norm_sq = new_max_norm_sq
        
        return selected
    
    # Strategy 1: Greedy by norm descending
    order1 = np.argsort(-norms_sq)
    sel1 = greedy_select(order1, deadline - 10)
    best_count = len(sel1)
    best_selected = sel1[:]
    
    # Strategy 2: Greedy by norm ascending (prefer low norm to keep threshold low)
    order2 = np.argsort(norms_sq)
    sel2 = greedy_select(order2, deadline - 10)
    if len(sel2) > best_count:
        best_count = len(sel2)
        best_selected = sel2[:]
    
    # Strategy 3: Random restarts
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        order = np.random.permutation(n_candidates)
        sel = greedy_select(order, deadline - 3)
        if len(sel) > best_count:
            best_count = len(sel)
            best_selected = sel[:]
    
    # Local search: try to add more vectors or swap
    if time.time() < deadline - 3 and best_count > 0:
        selected_arr = np.array(best_selected)
        max_norm_sq = np.max(np.sum(selected_arr * selected_arr, axis=1))
        
        # Try adding vectors not in selected
        selected_set = set(map(tuple, best_selected))
        added_any = True
        while added_any and time.time() < deadline - 2:
            added_any = False
            for idx in range(n_candidates):
                if time.time() > deadline - 2:
                    break
                v = candidates[idx]
                if tuple(v) in selected_set:
                    continue
                v_norm_sq = norms_sq[idx]
                new_max = max(max_norm_sq, v_norm_sq)
                
                if len(best_selected) == 0:
                    best_selected.append(v)
                    selected_set.add(tuple(v))
                    max_norm_sq = new_max
                    added_any = True
                    continue
                
                diffs = v - selected_arr
                dists_sq = np.sum(diffs * diffs, axis=1)
                if np.all(dists_sq >= new_max - 1e-9):
                    best_selected.append(v)
                    selected_set.add(tuple(v))
                    max_norm_sq = new_max
                    added_any = True
                    # Update selected_arr
                    selected_arr = np.array(best_selected)
                    break  # Restart loop since selected changed
        
        # Try swaps: replace one vector with two
        if time.time() < deadline - 2:
            for i in range(min(len(best_selected), 50)):
                if time.time() > deadline - 2:
                    break
                # Try removing best_selected[i] and adding others
                temp_selected = best_selected[:i] + best_selected[i+1:]
                if not temp_selected:
                    continue
                temp_arr = np.array(temp_selected)
                temp_max = np.max(np.sum(temp_arr * temp_arr, axis=1))
                temp_set = set(map(tuple, temp_selected))
                
                for idx in range(n_candidates):
                    if time.time() > deadline - 2:
                        break
                    v = candidates[idx]
                    if tuple(v) in temp_set:
                        continue
                    v_norm_sq = norms_sq[idx]
                    new_max = max(temp_max, v_norm_sq)
                    
                    diffs = v - temp_arr
                    dists_sq = np.sum(diffs * diffs, axis=1)
                    if np.all(dists_sq >= new_max - 1e-9):
                        # Check if we can add more after this
                        new_selected = temp_selected + [v]
                        if len(new_selected) > len(best_selected):
                            best_selected = new_selected
                            break
    
    result = np.array(best_selected, dtype=np.int64)
    
    # Verify validity
    if not _check_validity(result):
        # Fallback: just use norm sqrt(2) vectors
        fallback = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        fallback.append(v)
        return np.array(fallback, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
