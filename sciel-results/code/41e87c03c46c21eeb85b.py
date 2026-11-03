"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def _generate_all_vectors():
    """
    Generate a diverse pool of candidate vectors.
    We focus on shells with small norms to maximize packing density.
    Shell 1: 11 vectors (norm 1)
    Shell 2: 220 vectors (norm sqrt(2))
    Shell 3: 1716 vectors (norm sqrt(3))
    Shell 4: 11880 vectors (norm sqrt(4) = 2)
    
    Total ~ 13800 vectors.
    """
    vecs = []
    
    # Shell 1: e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 1
        vecs.append(v)
        
    # Shell 2: e_i - e_j
    for i in range(D):
        for j in range(D):
            if i == j: continue
            v = np.zeros(D, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            vecs.append(v)
            
    # Shell 3: e_i - e_j - e_k (and other sign combos? Let's stick to weight 3 with 1 pos, 2 neg or similar)
    # Actually, for kissing number, we want max inner product <= 1.
    # Let's generate all weight 3 vectors with entries in {1, -1}.
    # There are C(11,3) * 2^3 = 165 * 8 = 1320? No, C(11,3)=165. 165*8=1320.
    # Wait, C(11,3) is 165.
    # Let's just generate all weight 3.
    for comb in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    # Shell 4: Weight 4. C(11,4) * 2^4 = 330 * 16 = 5280.
    for comb in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    return np.array(vecs, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # 1. Generate candidate pool
    candidates = _generate_all_vectors()
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # 2. Initial Greedy Selection
    # We want a set S of indices such that for all u, v in S:
    # ||u - v||^2 >= max(||u||^2, ||v||^2)
    # ||u||^2 + ||v||^2 - 2<u,v> >= max(||u||^2, ||v||^2)
    # If ||u|| == ||v|| = R^2, then 2R^2 - 2<u,v> >= R^2 => R^2 >= 2<u,v> => <u,v> <= R^2/2.
    # If norms differ, say ||u||^2 >= ||v||^2, then ||u||^2 + ||v||^2 - 2<u,v> >= ||u||^2 => ||v||^2 >= 2<u,v>.
    
    # Let's use a unified constraint:
    # dist_sq(u, v) = ||u||^2 + ||v||^2 - 2<u,v>
    # We require dist_sq(u, v) >= max_norm_sq_current
    
    # Since we add vectors one by one, the "max_norm_sq" of the current set changes.
    # To simplify, let's sort candidates by norm squared (ascending).
    # Smaller norms are easier to pack.
    order = np.argsort(norms_sq)
    sorted_cands = candidates[order]
    sorted_norms_sq = norms_sq[order]
    
    selected_idx = []
    selected_vecs = []
    selected_norms_sq = []
    
    # Greedy: Iterate through sorted candidates.
    # For each candidate, check if it's compatible with all selected.
    # To speed up, we can keep track of the current max norm in the set.
    # However, the constraint depends on the specific pair.
    
    # Optimization: 
    # If we add a vector with norm R_new, it must satisfy:
    # For all u in S: ||u - v_new||^2 >= max(||u||^2, R_new^2)
    
    # Let's just do the full check. n_selected is likely < 1000.
    # n_cand is ~ 8000.
    # 8000 * 1000 * 11 flops is fine.
    
    for i in range(n_cand):
        if time.time() > deadline - 5: break
        
        v_new = sorted_cands[i]
        r_new_sq = sorted_norms_sq[i]
        
        if not selected_vecs:
            selected_idx.append(i)
            selected_vecs.append(v_new)
            selected_norms_sq.append(r_new_sq)
            continue
            
        # Check compatibility
        # We can vectorize this check against the current selected set.
        # dist_sq = ||v_new||^2 + ||u||^2 - 2 <v_new, u>
        # u_matrix is (n_sel, D)
        # dots = selected_vecs @ v_new  -> (n_sel,)
        # dists_sq = r_new_sq + selected_norms_sq - 2 * dots
        # max_norms = np.maximum(selected_norms_sq, r_new_sq)
        # valid = np.all(dists_sq >= max_norms)
        
        if len(selected_vecs) < 100:
             # Small set, just loop
             is_valid = True
             for j in range(len(selected_vecs)):
                 u = selected_vecs[j]
                 r_u_sq = selected_norms_sq[j]
                 dot = np.dot(u, v_new)
                 dist_sq = r_new_sq + r_u_sq - 2 * dot
                 req = max(r_new_sq, r_u_sq)
                 if dist_sq < req:
                     is_valid = False
                     break
             if is_valid:
                 selected_idx.append(i)
                 selected_vecs.append(v_new)
                 selected_norms_sq.append(r_new_sq)
        else:
             # Vectorized check
             u_mat = np.array(selected_vecs)
             dots = u_mat @ v_new
             dists_sq = r_new_sq + np.array(selected_norms_sq) - 2 * dots
             reqs = np.maximum(np.array(selected_norms_sq), r_new_sq)
             if np.all(dists_sq >= reqs):
                 selected_idx.append(i)
                 selected_vecs.append(v_new)
                 selected_norms_sq.append(r_new_sq)
                 
    # Initial set built.
    # Convert to list of indices in the original 'candidates' array?
    # No, selected_idx are indices into sorted_cands.
    # We need to work with indices into 'candidates' for the local search.
    # Let's map back.
    # Actually, it's easier to just work with the selected vectors directly in local search.
    # But we need to pick replacement candidates from 'candidates'.
    
    # Let's store the selected indices in the 'candidates' array.
    # sorted_cands[i] == candidates[order[i]]
    # So original_idx = order[sorted_idx]
    
    final_indices = [order[idx] for idx in selected_idx]
    
    # 3. Local Search (Hill Climbing)
    # Goal: Increase k (number of vectors) or improve "margin" to allow more additions.
    # Strategy: 
    # a) Try to add a vector from 'candidates' not in set.
    # b) If cannot add, try swapping one vector in set with one in candidates to allow a 2-for-1 or 1-for-1 improvement?
    #    Actually, simpler: Randomly remove a vector, try to add 2-3 others.
    #    Or: Pick a random candidate v_new. Find conflicts in current set.
    #    If number of conflicts is small, try to remove one conflict and add v_new?
    #    This is a "1-for-1" swap. It doesn't increase k.
    #    
    # To increase k, we need to find a "hole".
    # Heuristic: 
    # 1. Identify "tight" pairs in the current set.
    # 2. Try to replace a tight vector with a "looser" one?
    
    # Let's try a different local search: 
    # Simulated Annealing on the set of indices.
    # State: Set of indices.
    # Move: Remove one index, Add one index.
    # Cost: We want to maximize k. But k is fixed if we only swap.
    # We need to allow size changes.
    
    # Better Move: 
    # 1. Pick a random candidate v_add not in set.
    # 2. Check conflicts.
    # 3. If no conflicts, add it (k+1).
    # 4. If conflicts exist, pick a random conflict v_conf.
    # 5. Remove v_conf, add v_add. (k unchanged).
    #    Why would this help? It might change the geometry to allow future additions.
    #    This is "annealing" the geometry.
    
    # Let's implement this.
    
    # Data structures for speed
    # We need to quickly check if a candidate is compatible with the current set.
    # We can maintain the current set as a list of indices into 'candidates'.
    
    current_set = set(final_indices)
    current_vecs = candidates[list(current_set)]
    current_norms_sq = norms_sq[list(current_set)]
    
    # List of candidates NOT in current set
    # This list changes as we add/remove.
    # Instead of maintaining a list, we can just pick random indices from 0..n_cand-1 and check if in set.
    # To be more efficient, we can keep a list of "active" candidates.
    
    # Let's keep a list of all candidate indices.
    # We will iterate through them randomly.
    
    np.random.seed(42)
    
    # To speed up compatibility checks, we can precompute the Gram matrix?
    # n_cand is ~8000. 8000x8000 is 64M entries. int64 is 512MB. Too big.
    # We will compute dots on the fly.
    
    # Optimization: 
    # Store current set vectors in a matrix C (n_sel, D).
    # For a new candidate v, dots = C @ v.
    # This is O(n_sel * D).
    
    C = current_vecs
    C_norms_sq = current_norms_sq
    n_sel = len(C)
    
    # We need to track which indices are in the set to avoid duplicates.
    # And we need to be able to remove specific indices.
    
    # Let's use a list for the current set indices to allow fast removal?
    # Removal from list is O(n). n_sel ~ 1000. 1000 ops is fine.
    
    set_indices = list(current_set)
    
    # Map from index in 'candidates' to position in 'set_indices' for O(1) removal?
    # pos_map = {idx: i for i, idx in enumerate(set_indices)}
    # Updating pos_map on removal is tricky.
    # Given n_sel is small, linear search for removal is acceptable.
    
    def get_vec(idx):
        return candidates[idx]
    
    def is_compatible(v_idx):
        """Check if candidate v_idx is compatible with current set."""
        if v_idx in current_set:
            return True # Already there, trivially compatible with itself? No, we check against others.
        # Actually, if it's in the set, we don't need to add it.
        # This function is used to check if we CAN add it.
        
        v_new = candidates[v_idx]
        r_new_sq = norms_sq[v_idx]
        
        if n_sel == 0:
            return True
            
        # Vectorized check
        dots = C @ v_new
        dists_sq = r_new_sq + C_norms_sq - 2 * dots
        reqs = np.maximum(C_norms_sq, r_new_sq)
        
        return np.all(dists_sq >= reqs)
        
    # Main Loop
    while time.time() < deadline - 2:
        # 1. Try to add a random candidate
        v_add_idx = np.random.randint(0, n_cand)
        
        if v_add_idx not in current_set:
            if is_compatible(v_add_idx):
                # Add it
                current_set.add(v_add_idx)
                set_indices.append(v_add_idx)
                v_new = candidates[v_add_idx]
                r_new_sq = norms_sq[v_add_idx]
                
                # Update C and C_norms_sq
                C = np.vstack([C, v_new])
                C_norms_sq = np.append(C_norms_sq, r_new_sq)
                n_sel += 1
                continue # Restart loop, maybe we can add more
            
        # 2. If cannot add (or already in set), try a swap
        # Pick a random candidate to add
        v_add_idx = np.random.randint(0, n_cand)
        if v_add_idx in current_set:
            continue
            
        # Find conflicts with v_add_idx
        v_new = candidates[v_add_idx]
        r_new_sq = norms_sq[v_add_idx]
        
        if n_sel > 0:
            dots = C @ v_new
            dists_sq = r_new_sq + C_norms_sq - 2 * dots
            reqs = np.maximum(C_norms_sq, r_new_sq)
            conflict_mask = dists_sq < reqs
            
            if np.sum(conflict_mask) == 0:
                # Should have been caught above, but race condition?
                current_set.add(v_add_idx)
                set_indices.append(v_add_idx)
                C = np.vstack([C, v_new])
                C_norms_sq = np.append(C_norms_sq, r_new_sq)
                n_sel += 1
                continue
                
            # Pick a random conflict to remove
            conflict_positions = np.where(conflict_mask)[0]
            rem_pos = conflict_positions[np.random.randint(0, len(conflict_positions))]
            rem_idx = set_indices[rem_pos]
            
            # Perform swap: Remove rem_idx, Add v_add_idx
            # Check if this swap is "valid" in the sense that it doesn't break other constraints?
            # No, the swap is valid if v_add_idx is compatible with (Set - {rem_idx}).
            # We know v_add_idx conflicts with rem_idx.
            # Does it conflict with others? 
            # conflict_mask tells us ALL conflicts.
            # If we remove rem_idx, we must ensure v_add_idx is compatible with the rest.
            # So we need to check if there are OTHER conflicts.
            
            if np.sum(conflict_mask) > 1:
                # If there are multiple conflicts, removing just one might not be enough
                # unless the others are resolved? No, if it conflicts with A and B,
                # removing A doesn't make it compatible with B.
                # So a 1-for-1 swap only works if there is EXACTLY ONE conflict.
                # If there are multiple, we might need a 2-for-1 or similar.
                # For simplicity, let's only allow 1-for-1 swaps if there's exactly 1 conflict.
                # This limits moves but is safe.
                # Alternatively, we can try to remove ALL conflicts and add v_add?
                # That would be k - m + 1. If m > 1, we lose vectors.
                # We want to increase k. So we should avoid losing vectors.
                # So, only swap if num_conflicts == 1.
                pass
            else:
                # Exactly 1 conflict. Safe to swap.
                # Remove rem_idx
                current_set.remove(rem_idx)
                # Remove from set_indices
                # set_indices[rem_pos] = rem_idx
                # We need to remove it from the list.
                # Since we have the position, we can swap with last and pop?
                # But we need to maintain C and C_norms_sq consistency.
                
                # Let's just rebuild C and C_norms_sq after the swap?
                # n_sel is ~1000. Rebuilding is O(n_sel).
                # Doing this every swap might be slow if swaps are frequent.
                # But swaps are only allowed when a random candidate has exactly 1 conflict.
                # This might be rare.
                
                # Optimization: Update C in place?
                # C[rem_pos] = v_new?
                # C_norms_sq[rem_pos] = r_new_sq?
                # This works if we don't care about order.
                
                C[rem_pos] = v_new
                C_norms_sq[rem_pos] = r_new_sq
                set_indices[rem_pos] = v_add_idx
                current_set.add(v_add_idx)
                # k remains same.
                # This changes the geometry.
                
    # Final Result
    # Return the vectors corresponding to current_set
    result_indices = list(current_set)
    if not result_indices:
         # Fallback if something went wrong
         return np.array([np.zeros(D, dtype=np.int64) + 1], dtype=np.int64)
         
    return candidates[result_indices]

# EVOLVE-BLOCK-END
