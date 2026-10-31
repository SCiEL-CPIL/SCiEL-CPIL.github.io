import numpy as np
import itertools
import time

D = 11
DEADLINE = 130.0

def construct():
    start_time = time.time()
    
    # Generate all weight-4 vectors (four ±1's, rest 0)
    candidates = []
    
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    # Type 2: weight-1 vectors (±2 on one coordinate)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute dot products
    dots = candidates @ candidates.T
    
    # Constraint: dots[i,j] <= 2 for all i≠j (since all norm²=4)
    conflict = (dots > 2).copy()
    np.fill_diagonal(conflict, False)
    
    # Compute degrees
    degrees = conflict.sum(axis=1)
    
    best_selected = np.zeros(n, dtype=bool)
    best_count = 0
    
    np.random.seed(123)
    
    # Phase 1: Greedy with various orderings
    orderings = []
    orderings.append(np.argsort(degrees))  # degree ascending
    orderings.append(np.argsort(degrees)[::-1])  # degree descending
    
    for _ in range(10):
        orderings.append(np.random.permutation(n))
    
    # Degree with noise
    degree_with_noise = degrees + np.random.uniform(0, 1, n) * 0.3
    orderings.append(np.argsort(degree_with_noise))
    
    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        
        selected_mask = np.zeros(n, dtype=bool)
        count = 0
        
        for idx in order:
            if not np.any(selected_mask & conflict[idx]):
                selected_mask[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_selected = selected_mask.copy()
    
    # Phase 2: Add any non-selected vertex with no conflicts
    improved = True
    while improved and (time.time() - start_time < DEADLINE):
        improved = False
        non_selected = np.where(~best_selected)[0]
        for idx in non_selected:
            if time.time() - start_time > DEADLINE:
                break
            if not np.any(best_selected & conflict[idx]):
                best_selected[idx] = True
                best_count += 1
                improved = True
    
    # Phase 3: Swap-based local search
    for iteration in range(5):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        selected_indices = np.where(best_selected)[0]
        
        for sel_idx in selected_indices:
            if time.time() - start_time > DEADLINE:
                break
            
            saved = best_selected.copy()
            saved_count = best_count
            
            # Remove sel_idx
            best_selected[sel_idx] = False
            best_count -= 1
            
            # Greedily add non-conflicting vertices
            non_selected = np.where(~best_selected)[0]
            for non_idx in non_selected:
                if not np.any(best_selected & conflict[non_idx]):
                    best_selected[non_idx] = True
                    best_count += 1
            
            if best_count > saved_count:
                improved = True
            else:
                best_selected = saved
                best_count = saved_count
        
        if not improved:
            break
    
    result = candidates[best_selected]
    return result
