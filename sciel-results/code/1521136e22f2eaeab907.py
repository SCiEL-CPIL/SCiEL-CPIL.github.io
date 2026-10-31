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
    deadline = start_time + 110  # Safety margin
    
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
    
    # Greedy Independent Set with Random Restarts and Local Improvement
    np.random.seed(42)
    
    best_4_set = []
    best_4_size = 0
    
    # Number of restarts limited by time
    max_restarts = 200
    restarts_done = 0
    
    for attempt in range(max_restarts):
        if time.time() > deadline - 3:
            break
        restarts_done += 1
        
        # Greedy: pick node with min degree among remaining, with random tie-breaking
        remaining = np.ones(n4, dtype=bool)
        degrees = conflict.sum(axis=1).astype(np.int32)
        
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            # Get degrees of remaining
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            
            # Pick one of the min-degree nodes randomly
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and its neighbors
            to_remove = conflict[chosen_global].copy()
            to_remove[chosen_global] = True
            actual_remove = to_remove & remaining
            
            if np.any(actual_remove):
                # Update degrees
                deg_decrease = conflict[remaining][:, actual_remove].sum(axis=1)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
            
            # Time check
            if time.time() > deadline - 3:
                break
        
        # Local improvement: try to swap in a better node
        # Build a set for fast lookup
        selected_set = set(selected)
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            
            # For each selected node, try to remove it and add a non-conflicting node
            # that allows more nodes to be added
            for i in range(len(selected)):
                if time.time() > deadline - 3:
                    break
                    
                # Try removing selected[i]
                node_to_remove = selected[i]
                new_selected = selected[:i] + selected[i+1:]
                new_selected_set = set(new_selected)
                
                # Find nodes that don't conflict with new_selected
                # but conflict with the removed node's neighbors
                # Actually, we want to find a node that was excluded by node_to_remove
                # but doesn't conflict with new_selected
                
                # Get all nodes that conflict with node_to_remove
                conflicts_with_removed = np.where(conflict[node_to_remove])[0]
                
                # For each such node, check if it conflicts with new_selected
                for cand in conflicts_with_removed:
                    if cand in new_selected_set:
                        continue
                    
                    # Check if cand conflicts with any node in new_selected
                    conflicts_with_new = False
                    for ns in new_selected:
                        if conflict[cand, ns]:
                            conflicts_with_new = True
                            break
                    
                    if not conflicts_with_new:
                        # We can add cand
                        # But we also need to check if this leads to a better solution
                        # For now, just do a single swap
                        new_selected.append(cand)
                        new_selected_set.add(cand)
                        selected = new_selected
                        selected_set = new_selected_set
                        improved = True
                        break
                
                if improved:
                    break
        
        if len(selected) > best_4_size:
            best_4_size = len(selected)
            best_4_set = selected.copy()
            
        # Early termination if we've found a very good solution
        if best_4_size > 200:
            # Check if we can do better
            pass
    
    if best_4_size > best_size:
        best_set = candidates_4[best_4_set]
        best_size = best_4_size
        
    return best_set
