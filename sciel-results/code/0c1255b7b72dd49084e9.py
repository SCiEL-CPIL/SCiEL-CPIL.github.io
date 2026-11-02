import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    # Generate all weight-4 vectors (+/- 1 in exactly 4 positions)
    # Total: C(11,4) * 2^4 = 330 * 16 = 5280
    vectors = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    N = len(vectors)
    
    if time.time() > deadline:
        return vectors[:100]
    
    # Compute dot product matrix
    M = vectors @ vectors.T
    
    # Conflict matrix: conflict[i,j] = True if M[i,j] >= 3
    conflict = (M >= 3)
    
    # Symmetry breaking: keep only vectors where first non-zero entry is +1
    # This halves the set (for weight-4, exactly half have first non-zero = +1)
    sym_indices = []
    for i in range(N):
        # Find first non-zero coordinate
        nonzero_coords = np.nonzero(vectors[i])[0]
        if len(nonzero_coords) > 0:
            first_nonzero = nonzero_coords[0]
            if vectors[i, first_nonzero] == 1:
                sym_indices.append(i)
    
    sym_vectors = vectors[sym_indices]
    sym_N = len(sym_vectors)
    
    if time.time() > deadline:
        return sym_vectors[:100]
    
    # Conflict matrix for symmetry-broken set
    sub_conflict = conflict[np.ix_(sym_indices, sym_indices)]
    
    # Greedy min-degree selection
    # Start with all candidates available
    mask = np.ones(sym_N, dtype=bool)
    selected = []
    
    # Precompute degrees (number of conflicts)
    degrees = sub_conflict.sum(axis=1)
    
    while mask.any():
        if time.time() > deadline:
            break
        
        # Find candidate with minimum degree among available
        available_indices = np.where(mask)[0]
        min_deg_idx = available_indices[np.argmin(degrees[available_indices])]
        
        selected.append(min_deg_idx)
        
        # Remove min_deg_idx and all its conflicts from available
        mask[min_deg_idx] = False
        conflict_mask = sub_conflict[min_deg_idx] & mask
        mask[conflict_mask] = False
        
        # Update degrees for remaining available nodes
        if mask.any():
            # Recalculate degrees only for available nodes (approximation: subtract 1 from neighbors)
            # For speed, just recompute degrees for all remaining
            remaining = np.where(mask)[0]
            if len(remaining) > 0:
                degrees[remaining] = sub_conflict[remaining][:, remaining].sum(axis=1)
    
    if time.time() > deadline:
        return sym_vectors[selected]
    
    # Local search: 1-for-2 swaps
    # Try to improve by replacing one selected vector with two compatible ones
    S = set(selected)
    improved = True
    max_iter = 50
    
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        
        if time.time() > deadline:
            break
        
        S_list = list(S)
        # For each selected vector, try to replace with two compatible unselected vectors
        for u in S_list:
            if time.time() > deadline:
                break
            
            # Find all vectors that conflict with u
            conflicts_with_u = np.where(sub_conflict[u])[0]
            
            # Candidates: vectors not in S and not conflicting with u
            in_S = np.array([i in S for i in range(sym_N)])
            candidates = np.where(~in_S & ~conflicts_with_u)[0]
            
            if len(candidates) < 2:
                continue
            
            # Check if there's a compatible pair among candidates
            sub_conf = sub_conflict[np.ix_(candidates, candidates)]
            sub_conf_check = sub_conf.copy()
            np.fill_diagonal(sub_conf_check, True)
            
            # Find a pair with no conflict
            if (~sub_conf_check).any():
                row_idx, col_idx = np.where(~sub_conf_check)
                if len(row_idx) > 0:
                    r, c = row_idx[0], col_idx[0]
                    v1, v2 = candidates[r], candidates[c]
                    
                    # Replace u with v1 and v2
                    S.remove(u)
                    S.add(v1)
                    S.add(v2)
                    improved = True
                    break
        
        if time.time() > deadline:
            break
    
    result = sym_vectors[list(S)]
    return result
