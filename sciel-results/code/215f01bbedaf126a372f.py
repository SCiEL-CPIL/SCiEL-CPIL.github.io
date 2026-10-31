import numpy as np
import itertools

def construct():
    D = 11
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = 4
    
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    # Precompute neighbors as lists for faster iteration?
    # Or just use boolean masks.
    
    best_selected = []
    best_size = 0
    
    np.random.seed(42)
    
    # Run multiple restarts
    for attempt in range(100):
        remaining = np.ones(n, dtype=bool)
        degrees = conflict.sum(axis=1).copy()
        selected = []
        
        while remaining.any():
            # Get indices of remaining
            rem_indices = np.where(remaining)[0]
            
            if len(rem_indices) == 0:
                break
            
            # Find min degree among remaining
            # degrees is global array, but we only care about remaining
            # We can just take degrees[rem_indices]
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            
            # Get all with min degree
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            # Random choice
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and its neighbors from remaining
            # Neighbors of chosen_global
            neigh_mask = conflict[chosen_global]
            
            # Nodes to remove: chosen_global and its neighbors
            # But only those currently in remaining
            to_remove_mask = np.zeros(n, dtype=bool)
            to_remove_mask[chosen_global] = True
            to_remove_mask[neigh_mask] = True
            
            # Actual removal
            actual_remove = to_remove_mask & remaining
            
            # Update degrees of remaining nodes
            # For each node x in actual_remove, its neighbors in remaining lose 1 degree
            # We can do this vectorized?
            # degrees[remaining] -= conflict[actual_remove][:, remaining].sum(axis=0)
            
            # conflict is n x n boolean
            # conflict[actual_remove] is (num_remove, n)
            # We need sum of conflicts for each remaining node from the removed nodes
            
            if np.any(actual_remove):
                # Sum conflicts from removed nodes to all nodes
                # Only care about remaining nodes
                # conflict[actual_remove][:, remaining].sum(axis=0) gives the decrease in degree for each remaining node
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
    
    result = candidates[best_selected]
    return result
