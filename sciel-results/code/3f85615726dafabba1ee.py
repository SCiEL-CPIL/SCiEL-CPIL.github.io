"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Generate all nonzero integer vectors with entries in {-1, 0, 1}
    # We focus on weights 1-4 which gives a manageable candidate pool
    # Weight 1: 22, Weight 2: 220, Weight 3: 1320, Weight 4: 5280
    # Total: 6842 candidates
    
    candidates = []
    
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
                
    # Weight 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
                    
    # Weight 4
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
                        
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compute dot products matrix
    dots = np.dot(candidates, candidates.T)
    
    best_selected = []
    best_score = 0
    
    # We will iterate over possible M values (max norm squared)
    # For each M, we want to find a large set of vectors such that
    # for all distinct i, j in the set: ||v_i - v_j||^2 >= M
    # where M is the maximum squared norm in the set.
    
    # Possible M values from our candidates: 1, 2, 3, 4
    for M in [4, 3, 2, 1]:
        if time.time() > deadline - 15.0:
            break
            
        M_int = int(M)
        
        # Only consider candidates with norm_sq <= M
        mask = norms_sq <= M
        cand_indices = np.where(mask)[0]
        n_subset = len(cand_indices)
        
        if n_subset == 0:
            continue
        
        # Extract submatrix for this subset
        sub_dots = dots[np.ix_(cand_indices, cand_indices)]
        sub_norms = norms_sq[cand_indices]
        
        # Compute dist_sq for this subset
        dist_sq_sub = sub_norms[:, None] + sub_norms[None, :] - 2 * sub_dots
        np.fill_diagonal(dist_sq_sub, 0)
        
        # Conflict if dist_sq < M
        conflicts = (dist_sq_sub < M_int).astype(np.int32)
        
        # Use a greedy algorithm with multiple random restarts and orderings
        # to find a large independent set in the conflict graph
        
        # Precompute for efficiency: for each vertex, list of its neighbors
        # This helps in the greedy selection
        
        # Try multiple strategies:
        # 1. Greedy with norm-descending order
        # 2. Greedy with random order, multiple restarts
        # 3. Local search refinement
        
        # Strategy 1: Norm descending greedy
        selected_idx = _greedy_select_fast(candidates[cand_indices], norms_sq[cand_indices], 
                                           dist_sq_sub, M_int, start_time, deadline,
                                           order='norm_desc')
        if len(selected_idx) > best_score:
            best_score = len(selected_idx)
            best_selected = [cand_indices[i] for i in selected_idx]
        
        # Strategy 2: Multiple random restarts with random ordering
        n_restarts = 20
        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            if random.random() < 0.5:
                # Random order
                selected_idx = _greedy_select_fast(candidates[cand_indices], norms_sq[cand_indices], 
                                                   dist_sq_sub, M_int, start_time, deadline,
                                                   order='random')
            else:
                # Random with slight bias towards higher norms
                selected_idx = _greedy_select_fast(candidates[cand_indices], norms_sq[cand_indices], 
                                                   dist_sq_sub, M_int, start_time, deadline,
                                                   order='random_biased')
            if len(selected_idx) > best_score:
                best_score = len(selected_idx)
                best_selected = [cand_indices[i] for i in selected_idx]
                
        # Strategy 3: Local search refinement on the best found so far
        if best_selected and time.time() < deadline - 5.0:
            # Convert to local indices for this M
            best_local = [list(cand_indices).index(x) for x in best_selected]
            refined = _local_search(candidates[cand_indices], norms_sq[cand_indices],
                                    dist_sq_sub, M_int, best_local, start_time, deadline)
            if len(refined) > best_score:
                best_score = len(refined)
                best_selected = [cand_indices[i] for i in refined]
        
        # If we found a good solution with this M, check if we can do better
        # with a larger M (already trying M=4 first, so this is fine)
        # But we should also try to add more vectors if possible
        
    # Final local search on the best solution
    if best_selected and time.time() < deadline - 2.0:
        # Rebuild the context for the best M
        # Determine the M used
        max_norm_in_best = max(norms_sq[i] for i in best_selected)
        M_final = int(max_norm_in_best)
        
        mask = norms_sq <= M_final
        cand_indices = np.where(mask)[0]
        sub_dots = dots[np.ix_(cand_indices, cand_indices)]
        sub_norms = norms_sq[cand_indices]
        dist_sq_sub = sub_norms[:, None] + sub_norms[None, :] - 2 * sub_dots
        np.fill_diagonal(dist_sq_sub, 0)
        
        best_local = [list(cand_indices).index(x) for x in best_selected]
        refined = _local_search(candidates[cand_indices], norms_sq[cand_indices],
                                dist_sq_sub, M_final, best_local, start_time, deadline)
        if len(refined) > best_score:
            best_score = len(refined)
            best_selected = [cand_indices[i] for i in refined]
        
    if not best_selected:
        # Fallback: return a single vector
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        return np.array([v], dtype=np.int64)
        
    return candidates[best_selected]

def _greedy_select_fast(candidates_subset, norms_subset, dist_sq, M, start_time, deadline, order='random'):
    """
    Fast greedy selection using precomputed distance matrix.
    Returns indices into candidates_subset.
    """
    n = len(candidates_subset)
    if n == 0:
        return []
    
    # Determine ordering
    if order == 'norm_desc':
        order_indices = np.argsort(-norms_subset)
    elif order == 'random':
        order_indices = np.random.permutation(n)
    elif order == 'random_biased':
        # Bias towards higher norms but with randomness
        weights = norms_subset.astype(np.float64) + 0.1
        weights = weights / weights.sum()
        order_indices = np.random.choice(n, size=n, replace=False, p=weights)
    else:
        order_indices = np.arange(n)
    
    available = np.ones(n, dtype=bool)
    selected = []
    
    for idx in order_indices:
        if time.time() > deadline:
            break
        if not available[idx]:
            continue
        
        # Check if idx is compatible with all selected
        if not selected:
            selected.append(idx)
            available[idx] = False
            # Mark all neighbors as unavailable
            neighbors = np.where(dist_sq[idx, :] < M)[0]
            available[neighbors] = False
        else:
            # Check compatibility with all selected
            compatible = True
            for sel_idx in selected:
                if dist_sq[idx, sel_idx] < M:
                    compatible = False
                    break
            
            if compatible:
                selected.append(idx)
                available[idx] = False
                # Mark all neighbors as unavailable
                neighbors = np.where(dist_sq[idx, :] < M)[0]
                available[neighbors] = False
    
    return selected

def _local_search(candidates_subset, norms_subset, dist_sq, M, initial_selected, start_time, deadline, max_iters=500):
    """
    Local search to improve the independent set.
    Try to add vertices that fit, and if stuck, try swapping.
    """
    n = len(candidates_subset)
    selected = list(initial_selected)
    selected_set = set(selected)
    
    # Precompute: for each vertex, which selected vertices it conflicts with
    def get_conflicts(v):
        return [s for s in selected if dist_sq[v, s] < M]
    
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline - 2.0:
        improved = False
        iters += 1
        
        # Try to add a vertex
        added = False
        for v in range(n):
            if time.time() > deadline - 2.0:
                break
            if v in selected_set:
                continue
            
            conflicts = get_conflicts(v)
            if not conflicts:
                # Can add v
                selected.append(v)
                selected_set.add(v)
                improved = True
                added = True
                break
        
        if added:
            continue
        
        # Try swapping: remove one selected vertex and add two that fit
        # This is more complex, so let's try a simpler approach:
        # Try to find a pair (remove one, add one) that increases size
        # Or try to add a vertex by removing a conflicting one
        
        # For each unselected vertex, check if removing one of its conflicts
        # allows it to be added
        for v in range(n):
            if time.time() > deadline - 2.0:
                break
            if v in selected_set:
                continue
            
            conflicts = get_conflicts(v)
            if len(conflicts) == 1:
                # Removing the single conflict allows adding v, net zero change
                # But maybe v has higher norm? For now, skip
                pass
            elif len(conflicts) >= 2:
                # Try removing each conflict and see if v can be added
                for c in conflicts:
                    # Temporarily remove c
                    selected_set.discard(c)
                    selected.remove(c)
                    
                    # Check if v can be added
                    new_conflicts = get_conflicts(v)
                    if not new_conflicts:
                        # Add v
                        selected.append(v)
                        selected_set.add(v)
                        # Also try to add c back if possible
                        c_conflicts = get_conflicts(c)
                        if not c_conflicts:
                            selected.append(c)
                            selected_set.add(c)
                        improved = True
                        break
                    else:
                        # Put c back
                        selected.append(c)
                        selected_set.add(c)
    
    return selected

# EVOLVE-BLOCK-END
