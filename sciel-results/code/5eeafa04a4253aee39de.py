"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Strategy: 
    1. Generate all vectors with 4 non-zero entries (norm 2) and 8 non-zero entries (norm sqrt(8)).
    2. Use a randomized greedy algorithm to select a large subset satisfying the distance constraint.
    3. The constraint is ||u - v||^2 >= max_norm_sq.
       If max_norm_sq is 8 (from norm sqrt(8) vectors):
       - For two norm 2 vectors: 4 + 4 - 2 dot >= 8 => dot <= 0
       - For norm 2 and norm sqrt(8): 4 + 8 - 2 dot >= 8 => dot <= 2
       - For two norm sqrt(8): 8 + 8 - 2 dot >= 8 => dot <= 4
    """
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate candidates
    # Norm 2 vectors: 4 non-zero entries
    vecs_4 = []
    norms_4 = []
    for comb in combinations(range(D), 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        # Generate 16 sign variants
        for sign_mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for j, idx_val in enumerate(comb):
                if sign_mask & (1 << j):
                    v[idx_val] = 1
                else:
                    v[idx_val] = -1
            vecs_4.append(v)
            norms_4.append(4) # norm squared
    
    # Norm sqrt(8) vectors: 8 non-zero entries
    vecs_8 = []
    norms_8 = []
    for comb in combinations(range(D), 8):
        # Generate 256 sign variants
        for sign_mask in range(256):
            v = np.zeros(D, dtype=np.int64)
            for j, idx_val in enumerate(comb):
                if sign_mask & (1 << j):
                    v[idx_val] = 1
                else:
                    v[idx_val] = -1
            vecs_8.append(v)
            norms_8.append(8) # norm squared

    # Combine candidates
    all_vecs = np.array(vecs_4 + vecs_8, dtype=np.int64)
    all_norms = np.array(norms_4 + norms_8, dtype=np.int64)
    n_cands = len(all_vecs)
    
    # 2. Randomized Greedy MIS
    # We want to find a subset of indices such that for all i, j in subset:
    # norm_i + norm_j - 2 * dot(i, j) >= max(norm_i, norm_j) -- Wait, the condition is >= MAX norm in the whole set.
    # Since we have norm 8 vectors, the max norm is likely 8.
    # If we only select norm 4 vectors, max norm is 4.
    # But we want to maximize count, so we should include norm 8 vectors if possible.
    # Let's assume the final set will contain norm 8 vectors, so max_norm_sq = 8.
    # If the final set contains ONLY norm 4 vectors, max_norm_sq = 4.
    # We can try both:
    # Strategy A: Force max_norm_sq = 8. Only add vectors that satisfy condition w.r.t norm 8.
    # Strategy B: Force max_norm_sq = 4. Only add norm 4 vectors.
    # Strategy C: Adaptive. Start with norm 4, try to add norm 8. If adding norm 8 increases count, keep it.
    
    # Given the hint, adding norm 8 vectors increased count. So Strategy A or C is better.
    # Let's implement a greedy that tries to build a set satisfying max_norm_sq = 8.
    
    def _greedy_mis_max_norm_8(rng):
        selected_indices = []
        selected_vecs = []
        selected_norms = []
        
        # Shuffle candidates
        order = rng.permutation(n_cands)
        
        for idx in order:
            v = all_vecs[idx]
            norm_sq = all_norms[idx]
            
            # Check compatibility with all selected
            # Condition: ||v - u||^2 >= 8 for all u in selected
            # ||v||^2 + ||u||^2 - 2 v.dot(u) >= 8
            # norm_sq + norm_u - 2 dot >= 8
            
            if len(selected_vecs) == 0:
                selected_indices.append(idx)
                selected_vecs.append(v)
                selected_norms.append(norm_sq)
                continue
                
            # Vectorized check
            # selected_vecs is list of arrays, convert to matrix for speed?
            # For simplicity in this draft, we use a loop, but we should optimize.
            # Since n_cands is 47k, and selected size is ~500, 47k * 500 = 23.5M checks.
            # Each check is a dot product (11 ops). Total 250M ops.
            # This might be too slow in pure Python loops.
            
            # Optimization: Maintain a matrix of selected vectors.
            # But appending to a matrix is expensive.
            # Let's use a list and convert to array only when checking?
            # Or just use the list and numpy dot for each?
            
            compatible = True
            for i, u in enumerate(selected_vecs):
                dot = np.dot(v, u)
                norm_u = selected_norms[i]
                if norm_sq + norm_u - 2 * dot < 8:
                    compatible = False
                    break
            
            if compatible:
                selected_indices.append(idx)
                selected_vecs.append(v)
                selected_norms.append(norm_sq)
                
        return np.array(selected_vecs)

    # This loop is likely too slow. We need a faster method.
    # Let's try a different approach:
    # 1. Select a large set of norm-4 vectors satisfying dot <= 0.
    # 2. Then add norm-8 vectors.
    
    # Step 1: Maximize norm-4 vectors with dot <= 0.
    # This is a standard MIS problem on 5280 nodes.
    # We can use the previous greedy logic but with the condition dot <= 0.
    
    def _greedy_norm4_only(rng):
        # Use the 5280 norm-4 vectors
        n4 = len(vecs_4)
        vecs4_arr = np.array(vecs_4)
        
        # Pre-compute dot product matrix? 5280x5280 is 27M entries. 27M * 8 bytes = 216 MB.
        # This is feasible.
        dots = vecs4_arr @ vecs4_arr.T
        # Condition: dot <= 0
        # Conflict if dot > 0
        conflict = dots > 0
        
        # Greedy MIS
        available = np.ones(n4, dtype=bool)
        selected = np.zeros(n4, dtype=bool)
        
        # Degree
        deg = conflict.sum(axis=1)
        
        while available.any():
            avail_idx = np.where(available)[0]
            if len(avail_idx) == 0: break
            
            min_deg = deg[avail_idx].min()
            min_mask = (deg == min_deg) & available
            candidates = np.where(min_mask)[0]
            
            v = candidates[rng.integers(0, len(candidates))]
            selected[v] = True
            available[v] = False
            
            neighbors = np.where(conflict[v] & available)[0]
            if len(neighbors) > 0:
                available[neighbors] = False
                avail_after = np.where(available)[0]
                if len(avail_after) > 0:
                    deg[avail_after] -= conflict[avail_after, v]
                    
        return vecs4_arr[selected]

    best_result = None
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Run multiple tries
    for run in range(50):
        if time.time() > deadline - 10:
            break
            
        # Get a base set of norm-4 vectors
        base_vecs = _greedy_norm4_only(rng)
        
        # Try to add norm-8 vectors
        # We need to check compatibility of each norm-8 vector with all base_vecs.
        # And with other selected norm-8 vectors.
        
        # Let's try to add norm-8 vectors one by one in random order.
        # To speed up, we can pre-filter norm-8 vectors that are compatible with base_vecs.
        
        if len(base_vecs) == 0:
            continue
            
        # Vectorized check of base_vecs against all norm-8 vectors?
        # base_vecs is (k, 11). vecs_8 is (42240, 11).
        # Dots = base_vecs @ vecs_8.T -> (k, 42240)
        # This is k * 42240 * 11 ops.
        # If k=200, 200 * 42240 * 11 ~ 93M ops. Feasible.
        
        dots_base_8 = base_vecs @ np.array(vecs_8).T
        # Condition: base_norm + 8 - 2 * dot >= 8 => 4 + 8 - 2 dot >= 8 => dot <= 2
        # So we need all dots <= 2.
        # valid_8 = np.all(dots_base_8 <= 2, axis=0)
        
        # But we also need to check compatibility among norm-8 vectors themselves.
        # And we might want to select a subset of valid_8 that are mutually compatible.
        
        # Let's just greedily add valid_8 vectors.
        # First, identify which norm-8 vectors are compatible with the base set.
        valid_8_mask = np.all(dots_base_8 <= 2, axis=0)
        valid_8_indices = np.where(valid_8_mask)[0]
        
        # Now, from these valid_8 vectors, select a subset that are mutually compatible.
        # Condition: 8 + 8 - 2 dot >= 8 => dot <= 4.
        
        # We can use a greedy approach on these valid_8 vectors.
        # But 42240 is large.
        
        # Let's just pick random valid_8 vectors and check.
        # Or better, just return the base set for now if adding is complex.
        
        # Actually, let's just return the base set size for now to see the score.
        # The hint says adding norm-8 increased count.
        # So we MUST add them.
        
        # Let's try a simple greedy on the valid_8 vectors.
        # We need their dot products with each other.
        # We can't pre-compute 42240x42240.
        
        # Instead, we maintain a list of selected norm-8 vectors.
        selected_8 = []
        
        # Shuffle valid_8_indices
        rng.shuffle(valid_8_indices)
        
        vecs_8_arr = np.array(vecs_8)
        
        for idx in valid_8_indices:
            if time.time() > deadline - 5:
                break
                
            v8 = vecs_8_arr[idx]
            
            # Check against selected_8
            # Condition: dot <= 4
            if len(selected_8) == 0:
                selected_8.append(v8)
                continue
                
            # Check dot with all selected_8
            # To speed up, we can stop early if incompatible.
            compatible = True
            for u8 in selected_8:
                if np.dot(v8, u8) > 4:
                    compatible = False
                    break
                    
            if compatible:
                selected_8.append(v8)
                
        # Combine base and selected_8
        if len(selected_8) > 0:
            final_vecs = np.vstack([base_vecs, np.array(selected_8)])
        else:
            final_vecs = base_vecs
            
        size = len(final_vecs)
        if size > best_size:
            best_size = size
            best_result = final_vecs

    if best_result is None:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return best_result

# EVOLVE-BLOCK-END
