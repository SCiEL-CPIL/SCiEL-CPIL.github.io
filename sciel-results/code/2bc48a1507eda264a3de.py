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
    Shell 3: 1320 vectors (norm sqrt(3))
    Shell 4: 5280 vectors (norm sqrt(4) = 2)
    Shell 5: 17160 vectors (norm sqrt(5))
    Total ~ 24000 vectors.
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
            
    # Shell 3: Weight 3 vectors with entries in {1, -1}.
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
            
    # Shell 5: Weight 5. C(11,5) * 2^5 = 462 * 32 = 14784.
    # This increases the pool size significantly, potentially finding more compatible vectors.
    for comb in combinations(range(D), 5):
        for signs in product([1, -1], repeat=5):
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
    # Sort by norm squared (ascending) to prioritize smaller vectors which are easier to pack.
    order = np.argsort(norms_sq)
    sorted_cands = candidates[order]
    sorted_norms_sq = norms_sq[order]
    
    selected_idx = []
    selected_vecs = []
    selected_norms_sq = []
    
    # Greedy: Iterate through sorted candidates.
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
        if len(selected_vecs) < 100:
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
             u_mat = np.array(selected_vecs)
             dots = u_mat @ v_new
             dists_sq = r_new_sq + np.array(selected_norms_sq) - 2 * dots
             reqs = np.maximum(np.array(selected_norms_sq), r_new_sq)
             if np.all(dists_sq >= reqs):
                 selected_idx.append(i)
                 selected_vecs.append(v_new)
                 selected_norms_sq.append(r_new_sq)
                 
    final_indices = [order[idx] for idx in selected_idx]
    
    # 3. Local Search (Hill Climbing)
    # Strategy: 
    # 1. Try to add a vector from 'candidates' not in set.
    # 2. If cannot add, try swapping one vector in set with one in candidates to "anneal" geometry.
    #    Specifically, if a candidate has exactly 1 conflict, we can swap it in, removing the conflict.
    #    This keeps k constant but changes the set, potentially opening up space for future additions.
    
    current_set = set(final_indices)
    current_vecs = candidates[list(current_set)]
    current_norms_sq = norms_sq[list(current_set)]
    
    np.random.seed(42)
    
    C = current_vecs
    C_norms_sq = current_norms_sq
    n_sel = len(C)
    
    set_indices = list(current_set)
    
    def is_compatible(v_idx):
        if v_idx in current_set:
            return True
        v_new = candidates[v_idx]
        r_new_sq = norms_sq[v_idx]
        if n_sel == 0:
            return True
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
                current_set.add(v_add_idx)
                set_indices.append(v_add_idx)
                v_new = candidates[v_add_idx]
                r_new_sq = norms_sq[v_add_idx]
                C = np.vstack([C, v_new])
                C_norms_sq = np.append(C_norms_sq, r_new_sq)
                n_sel += 1
                continue
            
        # 2. If cannot add, try a swap
        v_add_idx = np.random.randint(0, n_cand)
        if v_add_idx in current_set:
            continue
            
        v_new = candidates[v_add_idx]
        r_new_sq = norms_sq[v_add_idx]
        
        if n_sel > 0:
            dots = C @ v_new
            dists_sq = r_new_sq + C_norms_sq - 2 * dots
            reqs = np.maximum(C_norms_sq, r_new_sq)
            conflict_mask = dists_sq < reqs
            
            if np.sum(conflict_mask) == 0:
                current_set.add(v_add_idx)
                set_indices.append(v_add_idx)
                C = np.vstack([C, v_new])
                C_norms_sq = np.append(C_norms_sq, r_new_sq)
                n_sel += 1
                continue
                
            # Only swap if there is EXACTLY ONE conflict.
            # This ensures the new vector is compatible with the rest of the set.
            if np.sum(conflict_mask) == 1:
                conflict_positions = np.where(conflict_mask)[0]
                rem_pos = conflict_positions[0]
                rem_idx = set_indices[rem_pos]
                
                # Perform swap
                current_set.remove(rem_idx)
                current_set.add(v_add_idx)
                C[rem_pos] = v_new
                C_norms_sq[rem_pos] = r_new_sq
                set_indices[rem_pos] = v_add_idx
                
    # Final Result
    result_indices = list(current_set)
    if not result_indices:
         return np.array([np.zeros(D, dtype=np.int64) + 1], dtype=np.int64)
         
    return candidates[result_indices]

# EVOLVE-BLOCK-END
