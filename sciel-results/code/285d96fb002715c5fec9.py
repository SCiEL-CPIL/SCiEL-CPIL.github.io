import numpy as np
import time
from itertools import combinations, product

def construct():
    """
    Construct nonzero integer vectors in R^11 maximizing count subject to:
    min_{u != v} ||u - v|| >= max_w ||w||.
    
    Strategy:
    1. Generate candidate vectors with small integer entries (weight 1, 2, 3).
    2. For each candidate weight (norm-squared group), determine the optimal 
       packing via maximum independent set on the conflict graph, where edges 
       connect vectors with distance^2 < norm^2 (using the max norm of the group).
       Use greedy heuristics (degree-based, weighted) plus simple local search.
    3. Return the best set found.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    # Generate all candidate vectors with entries in {-1, 0, 1} and weight <= 3
    # This covers D_11 minimal vectors (weight 2) and some weight 1, 3 vectors.
    
    all_candidates = []
    
    # Weight 1: single coordinate ±1
    for i in range(D):
        v = np.zeros(D, dtype=np.int8)
        v[i] = 1
        all_candidates.append(v)
        v[i] = -1
        all_candidates.append(v)
    
    # Weight 2: two coordinates ±1 (D_11 minimal vectors)
    for i in range(D):
        for j in range(i + 1, D):
            for si, sj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                v = np.zeros(D, dtype=np.int8)
                v[i] = si
                v[j] = sj
                all_candidates.append(v)
    
    # Weight 3: three coordinates ±1
    # This is C(11,3) * 8 = 165 * 8 = 1320 vectors.
    for indices in combinations(range(D), 3):
        for signs in product([-1, 1], repeat=3):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(indices, signs):
                v[idx] = s
            all_candidates.append(v)
    
    candidates = np.array(all_candidates, dtype=np.int8)
    n = len(candidates)
    
    if n == 0:
        return np.array([], dtype=np.int8).reshape(0, D)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # We'll try to find a valid set. The constraint is:
    # min_pairwise_dist >= max_norm
    # => for all u != v: ||u - v||^2 >= max_norm_sq
    # where max_norm_sq = max_w ||w||^2.
    
    # Since our candidates have norm^2 in {1, 2, 3}, max_norm_sq can be 1, 2, or 3.
    
    # Let's try each possible max_norm_sq in {1, 2, 3} and find the largest valid set.
    
    best_set = []
    best_score = 0
    
    for target_max_norm_sq in [1, 2, 3]:
        if time.time() > deadline - 5.0:
            break
            
        # Filter candidates with norm^2 <= target_max_norm_sq
        mask = norms_sq <= target_max_norm_sq
        valid_candidates = candidates[mask]
        n_valid = len(valid_candidates)
        
        if n_valid == 0:
            continue
        
        # Compute pairwise distance squared matrix (lower triangle)
        # dist2[i,j] = ||valid_candidates[i] - valid_candidates[j]||^2
        # = ||i||^2 + ||j||^2 - 2*i.dot(j)
        
        # Compute dot product matrix
        dots = valid_candidates @ valid_candidates.T
        norms_valid_sq = np.sum(valid_candidates * valid_candidates, axis=1)
        dist2 = norms_valid_sq[:, None] + norms_valid_sq[None, :] - 2 * dots
        
        # Conflict if dist2 < target_max_norm_sq
        # We want an independent set in the graph where edge exists if dist2 < target_max_norm_sq.
        
        # Greedy approach: start with all nodes, repeatedly remove nodes with most conflicts
        # until no conflicts remain. This is a maximal independent set via greedy removal.
        
        # But greedy removal might not be optimal. Let's try a few different strategies.
        
        # Strategy 1: Weighted greedy MIS - pick nodes with fewest conflicts among remaining
        selected = set()
        remaining = set(range(n_valid))
        
        # Precompute conflict count for each node
        conflict_counts = np.array([np.sum(dist2[i, j] < target_max_norm_sq for j in remaining if j != i) for i in range(n_valid)])
        
        # Actually, this is expensive. Let's use a simpler greedy: 
        # Start with empty set, add nodes that are compatible with all already selected.
        
        # Even simpler: use the fact that for small target_max_norm_sq, 
        # we can just check all pairs.
        
        # For target_max_norm_sq = 2 (D_11 case):
        # We know all 220 weight-2 vectors form a valid set.
        # Can we add weight-1 vectors? 
        # weight-1 vector e_i has norm^2=1. dist2(e_i, e_j+e_k) = 1+2-2*dot.
        # If j or k = i, dot=1 or 0. If dot=1, dist2=1 < 2. Conflict.
        # So e_i conflicts with e_i+e_j, e_i-e_j, e_j+e_i, e_j-e_i (4 vectors).
        # If we keep all 220, we can't add any e_i.
        # So for target=2, the 220 weight-2 vectors are optimal if we don't drop any.
        
        # For target_max_norm_sq = 3:
        # All weight 1, 2, 3 vectors with norm^2 <= 3.
        # Need dist2 >= 3 for all pairs.
        # This is a harder packing problem.
        
        # Let's just do a simple greedy for now.
        # Start with all valid candidates, iteratively remove nodes with highest conflict degree.
        
        active = np.ones(n_valid, dtype=bool)
        
        # Iterate: find node with most conflicts among active nodes, remove it
        for iteration in range(n_valid):
            if time.time() > deadline - 2.0:
                break
            if not np.any(active):
                break
            
            # Compute for each active node, how many active neighbors are in conflict
            active_idx = np.where(active)[0]
            if len(active_idx) == 0:
                break
            
            # Compute conflict counts only among active nodes
            sub_dist2 = dist2[np.ix_(active_idx, active_idx)]
            # dist2[i,j] < target_max_norm_sq means conflict
            conflict_mat = (sub_dist2 < target_max_norm_sq).astype(int)
            np.fill_diagonal(conflict_mat, 0)
            conf_counts = np.sum(conflict_mat, axis=1)
            
            if np.max(conf_counts) == 0:
                break
            
            # Remove the active node with highest conflict count
            remove_idx_local = np.argmax(conf_counts)
            remove_idx_global = active_idx[remove_idx_local]
            active[remove_idx_global] = False
            
        final_indices = np.where(active)[0]
        current_set = valid_candidates[final_indices]
        
        if len(current_set) > best_score:
            best_set = current_set
            best_score = len(current_set)
    
    # Also try the simple case: just all weight-2 vectors
    weight2_mask = norms_sq == 2
    weight2_vecs = candidates[weight2_mask]
    if len(weight2_vecs) > best_score:
        best_set = weight2_vecs
        best_score = len(weight2_vecs)
    
    if not isinstance(best_set, np.ndarray):
        best_set = np.array(best_set, dtype=np.int8)
    
    if len(best_set) == 0:
        return np.array([], dtype=np.int8).reshape(0, D)
    
    return best_set
