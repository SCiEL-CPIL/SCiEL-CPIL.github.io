import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # 1. Generate candidate vectors
    # We consider vectors with norm squared up to 16.
    max_norm_sq = 16
    candidates = []
    norms_sq = []
    
    # Helper to add vector
    def add_vec(v):
        ns = np.dot(v, v)
        if ns <= max_norm_sq:
            candidates.append(v)
            norms_sq.append(ns)

    # Generate all integer vectors with small norms
    # Iterate over possible supports and values
    # This is a bit slow if done naively, so we iterate over combinations of indices
    # and assign values.
    
    # Norm 1: e_i
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            add_vec(v)
            
    # Norm 2: e_i +/- e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                add_vec(v)
                
    # Norm 3: e_i +/- e_j +/- e_k
    for i, j, k in itertools.combinations(range(D), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    add_vec(v)
                    
    # Norm 4: 
    # Case 4a: 2e_i
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            add_vec(v)
            
    # Case 4b: e_i +/- e_j +/- e_k +/- e_l
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        add_vec(v)
                        
    # Case 4c: 2e_i +/- 2e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                add_vec(v)
                
    # Case 4d: 2e_i +/- e_j +/- e_k (Norm sq = 4+1+1=6? No, 2^2+1^2+1^2=6. > 4. Skip)
    # Wait, Norm 4 means norm squared = 4.
    # My generation above covers norm sq 1, 2, 3, 4.
    # 2e_i is norm sq 4.
    # e_i+e_j+e_k+e_l is norm sq 4.
    # 2e_i+2e_j is norm sq 8. 
    # Ah, I should filter by norm sq <= max_norm_sq.
    # The loop above adds them, and add_vec checks.
    
    # Case 4e: e_i+e_j+e_k with one entry 2? 
    # 2^2+1^2+1^2 = 6. > 4.
    # So only the ones listed are norm <= 4.
    
    # Let's add Norm 5, 6... up to 16?
    # This might be too many. 
    # Let's stick to Norm <= 4 for now, as 220 (Norm 2) is the baseline.
    # If we find better with Norm 4, we can expand.
    
    candidates = np.array(candidates)
    norms_sq = np.array(norms_sq)
    
    if len(candidates) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)

    best_set = []
    best_size = 0
    
    # Try different max norms
    unique_norms = sorted(set(norms_sq))
    
    for R_sq in unique_norms:
        if time.time() > deadline:
            break
            
        # Filter candidates with norm <= R_sq
        mask = norms_sq <= R_sq
        sub_vecs = candidates[mask]
        sub_norms = norms_sq[mask]
        
        if len(sub_vecs) == 0:
            continue
            
        # Compute pairwise distance squared
        # dist_sq[i,j] = ||v_i - v_j||^2
        # = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j
        # We need dist_sq >= R_sq for all i != j in selected set.
        # Conflict if dist_sq < R_sq.
        
        # Vectorized dot product
        # sub_vecs is (N, D)
        # dot = sub_vecs @ sub_vecs.T
        dot = sub_vecs @ sub_vecs.T
        dist_sq = sub_norms[:, None] + sub_norms[None, :] - 2 * dot
        
        # Conflict matrix: True if dist_sq < R_sq (and i != j)
        # Note: dist_sq[i,i] = 0 < R_sq. We ignore self-conflicts.
        conflict = dist_sq < R_sq
        np.fill_diagonal(conflict, False)
        
        # Greedy MIS
        # We want to select a subset of indices such that no two are connected in conflict graph.
        # This is Maximum Independent Set on the graph defined by 'conflict'.
        
        n = len(sub_vecs)
        if n == 0: continue
        
        # Heuristic: Greedy with random restarts
        # Sort by degree? Or just random.
        # Given time, a simple greedy with a few restarts might work.
        
        best_local = 0
        best_local_idx = []
        
        # Number of restarts depends on time available
        # Estimate time per restart: O(N^2) for selection? 
        # We can optimize selection using bitsets or boolean masks.
        
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random permutation
            perm = np.random.permutation(n)
            
            selected = []
            remaining = np.ones(n, dtype=bool)
            
            for idx in perm:
                if not remaining[idx]:
                    continue
                    
                # Check if idx conflicts with any selected
                # selected is list of indices.
                # Conflict with selected: conflict[idx, selected]
                # If any True, skip.
                
                # Optimization: maintain a boolean array 'can_select'
                # Initially all True.
                # When we select idx, we set can_select[conflict[idx, :]] = False.
                
                # This is O(N) per selection. Total O(N^2).
                # N is ~13k. N^2 is 169M. Might be slow in Python.
                
                # Let's try a faster greedy:
                # 1. Compute degrees in conflict graph.
                # 2. Pick node with min degree? (Min degree heuristic for MIS)
                #    Actually, for MIS, picking low degree nodes is good.
                
                # Let's use a simple approach:
                # Iterate through nodes in random order.
                # If node is not in conflict with any previously selected node, select it.
                
                # To speed up, we can use numpy boolean operations.
                # But doing it in a loop is slow.
                
                # Alternative:
                # If N is small (< 500), use exact or better heuristic.
                # If N is large, use a simpler heuristic.
                
                # Let's implement the simple loop but with early exit.
                # And limit restarts if N is large.
                
                if n > 5000:
                    # Too slow for many restarts. Just do 1 or 2.
                    if r > 0: break
                    
                selected_mask = np.zeros(n, dtype=bool)
                
                # Pre-compute conflict rows? No, just use the matrix.
                
                for i in perm:
                    if not remaining[i]:
                        continue
                    
                    # Check conflicts with already selected
                    # selected_indices = np.where(selected_mask)[0]
                    # This is slow.
                    
                    # Better:
                    # conflict_with_selected = conflict[i, selected_mask].any()
                    
                    if selected_mask.any():
                        if conflict[i, selected_mask].any():
                            continue
                            
                    # Select i
                    selected_mask[i] = True
                    # Mark conflicts as unavailable
                    # remaining[conflict[i, :]] = False
                    # This is O(N).
                    
                    # Update remaining
                    # We need to update 'remaining' for future checks?
                    # Actually, if we just check against 'selected_mask', we don't need 'remaining'
                    # except to skip already processed nodes?
                    # No, we iterate through perm. If i is already selected, skip.
                    # If i conflicts with selected, skip.
                    # So we just need selected_mask.
                    
                    # Wait, if we don't update 'remaining', we might check a node that is
                    # conflicting with a selected node, but we already handled that.
                    
                    # The logic:
                    # For each i in perm:
                    #   If i is already selected: continue
                    #   If i conflicts with any selected: continue
                    #   Select i
                    
                    # This works.
                    
                if selected_mask.sum() > best_local:
                    best_local = selected_mask.sum()
                    best_local_idx = np.where(selected_mask)[0]
                    
            if best_local > best_size:
                best_size = best_local
                best_set = sub_vecs[best_local_idx]
                
    if len(best_set) == 0:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
