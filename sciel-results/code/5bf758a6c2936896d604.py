import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    """
    D = 11
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit, leave 35s buffer)
    
    # Baseline: All vectors with exactly two non-zero entries (+/-1).
    # Norm^2 = 2. Max norm = sqrt(2).
    # Min distance between distinct such vectors is sqrt(2) (e.g., e1+e2 and e1+e3 -> diff e2-e3, norm sqrt(2)).
    # So all 220 are valid.
    candidates_2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates_2.append(v)
    
    baseline = np.array(candidates_2, dtype=np.int64)
    best_set = baseline
    best_size = len(baseline)
    
    # Try to find a larger set from Norm^2=4 vectors.
    # If we use only Norm^2=4, max norm = 2. Need dist^2 >= 4.
    
    candidates_4 = []
    # Type A: One entry +/-2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates_4.append(v)
            
    # Type B: Four entries +/-1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates_4.append(v)
            
    candidates_4 = np.array(candidates_4, dtype=np.int64)
    n4 = len(candidates_4)
    
    if time.time() > deadline - 20:
        return best_set
    
    # Compute conflict matrix for candidates_4
    # dist^2 = n_i + n_j - 2 dot
    # All norms are 4. dist^2 = 8 - 2 dot.
    # Conflict if dist^2 < 4 => 8 - 2 dot < 4 => 2 dot > 4 => dot > 2.
    # dot is integer. So conflict if dot >= 3.
    
    dots = candidates_4 @ candidates_4.T
    conflict = (dots >= 3)
    np.fill_diagonal(conflict, False)
    
    # Greedy Independent Set with Random Restarts and Local Search
    # We want a large set of nodes with no conflicts.
    
    np.random.seed(42)
    
    best_4_set = []
    
    # Number of restarts limited by time
    max_restarts = 50
    restarts_done = 0
    
    for attempt in range(max_restarts):
        if time.time() > deadline - 10:
            break
        restarts_done += 1
        
        remaining = np.ones(n4, dtype=bool)
        # Initial degrees
        degrees = conflict.sum(axis=1).astype(np.int32).copy()
        
        selected = []
        
        # Greedy: pick node with min degree, remove it and its neighbors
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            # Get degrees of remaining
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            
            # Pick one of the min-degree nodes
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and neighbors
            to_remove = conflict[chosen_global].copy()
            to_remove[chosen_global] = True
            actual_remove = to_remove & remaining
            
            if np.any(actual_remove):
                # Update degrees
                # deg_new[r] = deg_old[r] - count(neighbors of r in actual_remove)
                deg_decrease = conflict[remaining][:, actual_remove].sum(axis=1)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
            
            # Time check
            if time.time() > deadline - 10:
                break
                
        # Local Search: Try to improve the current solution by swapping
        # This is a simple 1-swap local search
        selected_set = set(selected)
        improved = True
        while improved:
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 10:
                    break
                v1 = selected[i]
                # Find a node not in selected that conflicts with v1 but has fewer conflicts overall
                # Actually, we want to find a node v2 not in selected such that:
                # 1. v2 does not conflict with any node in selected except possibly v1
                # 2. v1 conflicts with more nodes than v2 does with the rest of selected
                # This is complex. A simpler approach:
                # Try to remove v1 and add a non-selected node that has 0 conflicts with the new set.
                
                # Let's try a different local search: 
                # For each node in selected, try to swap it with a non-selected node
                # that has the same or fewer conflicts with the rest of the set.
                
                rest = [v for j, v in enumerate(selected) if j != i]
                
                # Count conflicts of v1 with rest
                conflicts_v1 = sum(1 for v in rest if conflict[v1, v])
                
                # Find candidates to swap in
                # A candidate v2 must:
                # - Not be in selected
                # - Have 0 conflicts with rest
                # - Have <= conflicts_v1 conflicts with rest (this is always 0 if it has 0 conflicts)
                # - Not conflict with v1 (actually, we're removing v1, so this doesn't matter)
                
                # Let's just find any non-selected node with 0 conflicts with rest
                # This is expensive. Let's limit the search.
                
                # For efficiency, let's just try to add nodes if possible, which we do after the main loop.
                # For now, let's skip complex swaps and rely on the greedy + add step.
                
                pass
                
        if len(selected) > len(best_4_set):
            best_4_set = selected
            
    if len(best_4_set) > best_size:
        best_set = candidates_4[best_4_set]
        best_size = len(best_4_set)
        
    # Try to improve with local search (add-based)
    # For each non-selected vector, try adding it if it has no conflicts with the current set
    
    if best_size > 0 and len(best_4_set) > 0:
        # Convert best_4_set to a boolean mask
        mask = np.zeros(n4, dtype=bool)
        mask[best_4_set] = True
        
        # Repeat until no more can be added
        while True:
            # Compute the set of conflicts with the current selected set
            # conflict_with_selected[i] = number of selected nodes that conflict with i
            conflict_with_selected = (conflict[:, mask].sum(axis=1))
            
            # Find non-selected nodes with 0 conflicts
            addable = (~mask) & (conflict_with_selected == 0)
            
            if not np.any(addable):
                break
            
            new_indices = np.where(addable)[0]
            for idx in new_indices:
                mask[idx] = True
            
            best_4_set = np.where(mask)[0].tolist()
            best_size = len(best_4_set)
            best_set = candidates_4[best_4_set]
            
            if time.time() > deadline - 5:
                break
    
    return best_set
