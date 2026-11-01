import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Generate all candidates with squared norm 4
    # Type 1: 4 non-zero entries, each +/- 1
    candidates = []
    indices = list(range(D))
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates.append(v)
            
    # Type 2: 1 non-zero entry, value +/- 2
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        candidates.append(v)
        v[i] = -2
        candidates.append(v)
        
    candidates = np.array(candidates)
    n = len(candidates)
    
    # Precompute all pairwise dot products
    dots = candidates @ candidates.T
    np.fill_diagonal(dots, 0)
    
    # Conflict: dot > 2 (since we need min distance >= max norm = 2)
    # For norm-4 vectors, distance^2 = 8 - 2*dot >= 4 => dot <= 2
    conflict = dots > 2
    np.fill_diagonal(conflict, False)
    
    # Build compatibility graph
    compatible = ~conflict
    np.fill_diagonal(compatible, False)
    
    # Use a more sophisticated approach:
    # 1. Start with a good initial solution using greedy
    # 2. Improve using local search with swaps
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy by degree in compatibility graph
    degrees = compatible.sum(axis=1)
    
    if time.time() < deadline - 20:
        order = np.argsort(-degrees)
        selected = []
        for idx in order:
            if time.time() > deadline - 10:
                break
            if len(selected) == 0:
                selected.append(idx)
                continue
            # Check compatibility with all selected
            if not np.any(conflict[idx, selected]):
                selected.append(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Strategy 2: Multiple randomized greedy restarts
    num_restarts = 50
    rng = np.random.RandomState(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random ordering with some bias toward high-degree nodes
        weights = degrees + 1
        weights = weights / weights.sum()
        perm = rng.choice(n, size=n, replace=False, p=weights)
        
        selected = []
        for idx in perm:
            if time.time() > deadline - 10:
                break
            if len(selected) == 0:
                selected.append(idx)
                continue
            if not np.any(conflict[idx, selected]):
                selected.append(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Strategy 3: Local search improvement
    if best_set and time.time() < deadline - 30:
        selected_set = set(best_set)
        current_size = len(selected_set)
        no_improvement_count = 0
        
        while no_improvement_count < 5 and time.time() < deadline - 10:
            improved = False
            
            # Try to add any compatible vector
            for idx in range(n):
                if time.time() > deadline - 10:
                    break
                if idx in selected_set:
                    continue
                if not np.any(conflict[idx, list(selected_set)]):
                    selected_set.add(idx)
                    improved = True
            
            # Try swap: remove one selected, add two non-selected
            if not improved:
                sel_list = list(selected_set)
                for i in range(len(sel_list)):
                    if time.time() > deadline - 10:
                        break
                    candidate_remove = sel_list[i]
                    temp_set = set(selected_set)
                    temp_set.remove(candidate_remove)
                    
                    added = 0
                    for j in range(n):
                        if time.time() > deadline - 10:
                            break
                        if j in temp_set:
                            continue
                        if not np.any(conflict[j, list(temp_set)]):
                            temp_set.add(j)
                            added += 1
                            if added >= 2:
                                break
                    
                    if added >= 2:
                        selected_set = temp_set
                        improved = True
                        break
            
            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
        
        if len(selected_set) > best_size:
            best_size = len(selected_set)
            best_set = list(selected_set)
    
    # Convert to result array
    if best_set:
        result = candidates[best_set]
    else:
        # Fallback: return a single valid vector
        result = np.array([[1,1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    return result
