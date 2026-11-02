"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate a diverse pool of candidate vectors
    # We include vectors of various norms to allow for a denser packing.
    # Norm 1: 22 vectors
    # Norm 2: 220 vectors
    # Norm 3: 1980 vectors
    # Norm 4: 5522 vectors (4 single signs + 1 double sign (2) + 2 double signs (2,2))
    # Norm 5: 22000 vectors (5 single signs)
    # Total candidates ~ 30,000. This is large but manageable with vectorized ops if we are careful.
    # However, 30k x 30k matrix is 900M entries, too big.
    # We will use a smaller pool or a smarter selection.
    # Let's stick to Norms 1, 2, 3, 4 for a pool of ~7700 vectors.
    
    candidates = []
    
    # Norm 1
    for i in range(D):
        v1 = np.zeros(D, dtype=np.int64); v1[i] = 1; candidates.append(v1)
        v2 = np.zeros(D, dtype=np.int64); v2[i] = -1; candidates.append(v2)
        
    # Norm 2
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si; v[j] = sj
                    candidates.append(v)
                    
    # Norm 3
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si; v[j] = sj; v[k] = sk
                            candidates.append(v)
                            
    # Norm 4 (Type 1: four 1/-1)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            for sk in [1, -1]:
                                for sl in [1, -1]:
                                    v = np.zeros(D, dtype=np.int64)
                                    v[i] = si; v[j] = sj; v[k] = sk; v[l] = sl
                                    candidates.append(v)
                                    
    # Norm 4 (Type 2: one 2)
    for i in range(D):
        for si in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = si * 2
            candidates.append(v)
            
    # Norm 4 (Type 3: two 2s)
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si * 2; v[j] = sj * 2
                    candidates.append(v)

    candidates = np.array(candidates)
    n_cand = len(candidates)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # 2. Greedy Initialization
    # Sort candidates by norm squared ascending? Or random?
    # Random restarts with different orderings.
    best_set_indices = []
    best_count = 0
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() > deadline - 100: break
        perm = rng.permutation(n_cand)
        curr_set = []
        curr_norms = []
        for idx in perm:
            v = candidates[idx]
            if not curr_set:
                curr_set.append(idx)
                curr_norms.append(norms_sq[idx])
                continue
            
            sel_vecs = candidates[curr_set]
            sel_norms = np.array(curr_norms)
            dots = sel_vecs @ v
            limits = np.minimum(norms_sq[idx], sel_norms)
            if np.all(2 * dots <= limits):
                curr_set.append(idx)
                curr_norms.append(norms_sq[idx])
        
        if len(curr_set) > best_count:
            best_count = len(curr_set)
            best_set_indices = curr_set.copy()
            
    # 3. Hungarian Algorithm Local Search
    # We treat the problem as: Given a set S of K vectors, try to swap one vector out and one in.
    # But a better approach: 
    # Identify "bottleneck" pairs. 
    # Or, more simply, use the Hungarian algorithm to find the best permutation of a subset?
    # Actually, a common heuristic for kissing numbers is to start with a known configuration 
    # and then perform local searches.
    
    # Let's try a "repair" strategy.
    # Start with the best greedy set.
    current_set = best_set_indices.copy()
    
    # Precompute compatibility matrix for the current set?
    # If |current_set| is ~500, the matrix is 500x500.
    # But we want to swap with candidates outside the set.
    
    # Let's define a cost matrix for swapping.
    # For each i in current_set, and each j in candidates (not in current_set):
    # Cost(i, j) = 0 if swapping i for j increases the "slack" or maintains validity, else infinity.
    # This is hard to define globally.
    
    # Alternative: Simulated Annealing with Hungarian-guided moves?
    # Let's stick to a robust local search:
    # 1. Try to add a vector. If conflict, find the vector in the set that conflicts the most.
    # 2. Remove it.
    # 3. If the new set is valid, accept.
    
    # We already have a greedy set. Let's try to improve it by trying to add more vectors.
    # We will iterate through all candidates and try to insert them.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        curr_set_arr = np.array(current_set)
        curr_vecs = candidates[curr_set_arr]
        curr_norms = norms_sq[curr_set_arr]
        
        # Create a mask of candidates not in current_set
        in_set_mask = np.zeros(n_cand, dtype=bool)
        in_set_mask[curr_set_arr] = True
        out_set_indices = np.where(~in_set_mask)[0]
        
        # For each candidate outside, check if it can be added.
        # To speed this up, we can batch process.
        # But n_out is ~7000, n_in is ~500.
        # 7000 * 500 * 11 ops = 38M ops. Fast in numpy.
        
        # Let's check all out-set candidates against the current set.
        out_vecs = candidates[out_set_indices]
        out_norms = norms_sq[out_set_indices]
        
        # Compute dots: (n_out, D) @ (D, n_in) -> (n_out, n_in)
        dots = out_vecs @ curr_vecs.T
        # limits: (n_out, 1) and (1, n_in) -> (n_out, n_in)
        limits = np.minimum(out_norms[:, None], curr_norms[None, :])
        
        # conflicts: (n_out, n_in) boolean matrix
        conflicts = 2 * dots > limits
        
        # For each candidate, find if it has 0 conflicts.
        # If yes, add it.
        zero_conflict_mask = np.all(~conflicts, axis=1)
        zero_conflict_indices = np.where(zero_conflict_mask)[0]
        
        if len(zero_conflict_indices) > 0:
            # Add all compatible candidates? 
            # Adding one might make others incompatible.
            # Greedily add them one by one, re-checking?
            # Or add the one with the largest norm?
            # Let's add the first one found, set improved=True, and break to re-evaluate.
            # Or add all if they are mutually compatible?
            # Let's just add one to be safe and fast.
            idx_to_add = zero_conflict_indices[0]
            current_set.append(out_set_indices[idx_to_add])
            improved = True
            # Continue loop to re-evaluate
            continue
            
        # If no vector can be added directly, try swaps.
        # For each candidate j outside, find the vector i inside that conflicts with it.
        # If we remove i, is j compatible with the rest?
        # This is O(n_out * n_in * n_in) which is too slow.
        
        # Heuristic: Try to swap a random conflicting pair.
        # Pick a random candidate j outside.
        # Find its conflicts in the set.
        # Pick a random conflict i.
        # Remove i, check if j is compatible with rest.
        
        if len(out_set_indices) > 0:
            # Sample a few random candidates to try swaps
            n_trials = min(100, len(out_set_indices))
            trial_indices = rng.choice(len(out_set_indices), n_trials, replace=False)
            
            for t_idx in trial_indices:
                j_global = out_set_indices[t_idx]
                v_j = candidates[j_global]
                n_j = norms_sq[j_global]
                
                # Find conflicts with current set
                dots_j = curr_vecs @ v_j
                limits_j = np.minimum(n_j, curr_norms)
                conflicts_j = 2 * dots_j > limits_j
                
                if not np.any(conflicts_j):
                    # Should have been caught above, but safety check
                    current_set.append(j_global)
                    improved = True
                    break
                
                conflict_indices_in_set = np.where(conflicts_j)[0]
                if len(conflict_indices_in_set) == 0:
                    continue
                    
                # Pick a random conflict to remove
                idx_conf = conflict_indices_in_set[rng.integers(0, len(conflict_indices_in_set))]
                
                # Temporarily remove
                removed_idx = current_set[idx_conf]
                current_set.pop(idx_conf)
                
                # Check compatibility with rest
                rest_set = np.array(current_set)
                if len(rest_set) == 0:
                    current_set.append(j_global)
                    improved = True
                    break
                    
                rest_vecs = candidates[rest_set]
                rest_norms = norms_sq[rest_set]
                
                dots_rest = rest_vecs @ v_j
                limits_rest = np.minimum(n_j, rest_norms)
                
                if np.all(2 * dots_rest <= limits_rest):
                    current_set.append(j_global)
                    improved = True
                    break
                else:
                    # Restore
                    current_set.insert(idx_conf, removed_idx)
                    
    # Final check and return
    if not current_set:
        return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)
        
    final_vecs = candidates[np.array(current_set)]
    return final_vecs

# EVOLVE-BLOCK-END
