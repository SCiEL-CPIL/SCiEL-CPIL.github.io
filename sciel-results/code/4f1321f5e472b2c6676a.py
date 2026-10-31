import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute conflict sets as boolean arrays for fast operations
    # conflicts[i] is a boolean array of length N4
    
    # Strategy: Multiple greedy passes with different orderings, then vectorized local search
    
    best_selected = None
    best_count = 0
    
    # Helper: greedily build independent set from an ordering
    def greedy_from_order(order, conflicts, N4):
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            alive[i] = False
            alive[conflicts[i]] = False
        return selected
    
    # Helper: extend independent set by adding all compatible vertices (vectorized)
    def extend_independent_set(selected, conflicts, N4, max_iters=200):
        for _ in range(max_iters):
            # Find all non-selected vertices that don't conflict with any selected vertex
            non_sel = np.where(~selected)[0]
            if len(non_sel) == 0:
                break
            # A vertex v is compatible if no selected vertex conflicts with it
            # i.e., for all u in selected, conflicts[v, u] == False
            # Equivalently: conflicts[non_sel][:, selected] has no True entries
            compat = ~np.any(conflicts[non_sel][:, selected], axis=1)
            if not np.any(compat):
                break
            add_idx = non_sel[compat]
            selected[add_idx] = True
        return selected
    
    # Strategy 1: Greedy minimum degree
    rng = np.random.default_rng(42)
    alive = np.ones(N4, dtype=bool)
    selected = np.zeros(N4, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int64)
    
    while np.any(alive):
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        alive_degrees = degrees[alive_indices]
        min_deg = np.min(alive_degrees)
        cands = alive_indices[alive_degrees == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected[chosen] = True
        to_remove = np.where(conflicts[chosen] | (np.arange(N4) == chosen))[0]
        alive[to_remove] = False
        remaining = np.where(alive)[0]
        if len(remaining) > 0:
            degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
    
    selected = extend_independent_set(selected, conflicts, N4)
    if selected.sum() > best_count:
        best_count = selected.sum()
        best_selected = selected.copy()
    
    # Strategy 2: Random orderings
    for restart in range(30):
        if time.time() > deadline - 20:
            break
        rng2 = np.random.default_rng(1000 + restart)
        order = rng2.permutation(N4)
        selected = greedy_from_order(order, conflicts, N4)
        selected = extend_independent_set(selected, conflicts, N4)
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Strategy 3: Degree-based ordering (sort by degree, pick min first)
    degree_order = np.argsort(conflicts.sum(axis=1))
    selected = greedy_from_order(degree_order, conflicts, N4)
    selected = extend_independent_set(selected, conflicts, N4)
    if selected.sum() > best_count:
        best_count = selected.sum()
        best_selected = selected.copy()
    
    # Strategy 4: Local search with vectorized swap operations
    if best_selected is not None:
        selected = best_selected.copy()
        rng_ls = np.random.default_rng(77)
        
        # Precompute for each vertex: which selected vertices conflict with it
        # and which non-selected vertices it conflicts with
        
        improved = True
        max_rounds = 50
        round_num = 0
        while improved and round_num < max_rounds:
            if time.time() > deadline - 10:
                break
            improved = False
            round_num += 1
            
            sel_idx = np.where(selected)[0]
            non_sel_idx = np.where(~selected)[0]
            
            if len(non_sel_idx) == 0 or len(sel_idx) == 0:
                break
            
            # For each non-selected vertex, count how many selected vertices it conflicts with
            # conflict_count[v] = number of selected u such that conflicts[v, u]
            conflict_counts = conflicts[non_sel_idx][:, sel_idx].sum(axis=1)
            
            # Vertices with conflict_count == 0 can be added freely (shouldn't exist after extend)
            # Vertices with conflict_count == 1: try swapping
            single_conflict_mask = conflict_counts == 1
            if np.any(single_conflict_mask):
                single_idx = non_sel_idx[single_conflict_mask]
                
                # For each such vertex v, find the one selected vertex u it conflicts with
                # Try: remove u, add v, then extend
                best_gain = 0
                best_swap = None
                
                # Vectorize: for each v in single_idx, find its conflicting selected u
                # conflicts[v, sel_idx] has exactly one True
                v_conflict_sel = conflicts[single_idx][:, sel_idx]  # (len(single_idx), len(sel_idx))
                # Find the index of the single True in each row
                u_positions = np.argmax(v_conflict_sel, axis=1)  # position in sel_idx
                u_indices = sel_idx[u_positions]
                
                # Try all swaps in batches - but this is expensive
                # Instead, sample if too many
                if len(single_idx) > 500:
                    sample = rng_ls.choice(len(single_idx), size=500, replace=False)
                    single_idx = single_idx[sample]
                    u_indices = u_indices[sample]
                
                for i in range(len(single_idx)):
                    if time.time() > deadline - 10:
                        break
                    v = single_idx[i]
                    u = u_indices[i]
                    
                    # Tentative swap
                    selected[u] = False
                    selected[v] = True
                    
                    # Extend
                    new_selected = extend_independent_set(selected.copy(), conflicts, N4, max_iters=100)
                    gain = new_selected.sum() - selected.sum()
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (u, v, new_selected.copy())
                    
                    # Revert
                    selected[u] = True
                    selected[v] = False
                
                if best_swap is not None and best_gain > 0:
                    u, v, new_sel = best_swap
                    selected = new_sel
                    improved = True
            
            # Also try: remove a selected vertex that blocks many others
            if not improved and len(sel_idx) > 0:
                # For each selected vertex u, count how many non-selected vertices it blocks
                # i.e., non-selected v such that conflicts[v, u] == True and v has no other selected conflicts
                block_counts = np.zeros(len(sel_idx), dtype=np.int64)
                
                # For each non-selected v with conflict_count == 1, it's blocked by exactly one selected
                # Count how many such v are blocked by each selected u
                single_mask = conflict_counts == 1
                if np.any(single_mask):
                    single_v = non_sel_idx[single_mask]
                    v_conf_sel = conflicts[single_v][:, sel_idx]
                    u_pos = np.argmax(v_conf_sel, axis=1)
                    # Count occurrences
                    for pos in u_pos:
                        block_counts[pos] += 1
                
                if np.max(block_counts) > 0:
                    max_block_pos = np.argmax(block_counts)
                    u = sel_idx[max_block_pos]
                    if block_counts[max_block_pos] >= 2:  # Only worth it if we block >= 2
                        selected[u] = False
                        new_selected = extend_independent_set(selected.copy(), conflicts, N4, max_iters=100)
                        if new_selected.sum() > selected.sum():
                            selected = new_selected
                            improved = True
                        else:
                            selected[u] = True
        
        if selected.sum() > best_count:
            best_count = selected.sum()
            best_selected = selected.copy()
    
    # Final extension
    if best_selected is not None:
        selected = extend_independent_set(best_selected.copy(), conflicts, N4, max_iters=500)
        if selected.sum() > best_count:
            best_count = selected.sum()
            best_selected = selected.copy()
    
    result_v4 = vectors_v4[best_selected]
    
    # Add all V1 vectors (they're all compatible with V4 and each other)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    # Verify the constraint: min pairwise distance >= max norm
    # For safety, verify and remove any violating vectors
    if len(final_vectors) > 1:
        norms = np.sqrt(np.sum(final_vectors.astype(np.float64)**2, axis=1))
        max_norm = np.max(norms)
        max_norm_sq = max_norm ** 2
        
        # Compute all pairwise distances efficiently in batches
        n_final = len(final_vectors)
        valid = np.ones(n_final, dtype=bool)
        
        batch_size = 500
        for i in range(0, n_final, batch_size):
            if time.time() > deadline - 3:
                break
            batch = final_vectors[i:i+batch_size]
            if len(batch) == 0:
                break
            # Distance squared: d(a,b)^2 = |a|^2 + |b|^2 - 2*a.b
            dots_batch = batch @ final_vectors.T
            norms_sq_batch = np.sum(batch.astype(np.float64)**2, axis=1)
            all_norms_sq = np.sum(final_vectors.astype(np.float64)**2, axis=1)
            dist_sq = norms_sq_batch[:, None] + all_norms_sq[None, :] - 2 * dots_batch
            
            # Check each vector in batch against all valid vectors
            for r in range(len(batch)):
                idx_in_final = i + r
                if not valid[idx_in_final]:
                    continue
                d_sq = dist_sq[r, :]
                d_sq[idx_in_final] = np.inf
                # Find any valid vector with distance < max_norm
                violating = np.where((d_sq < max_norm_sq - 1e-6) & valid)[0]
                if len(violating) > 0:
                    valid[idx_in_final] = False
                    valid[violating] = False
        
        final_vectors = final_vectors[valid]
    
    return final_vectors
