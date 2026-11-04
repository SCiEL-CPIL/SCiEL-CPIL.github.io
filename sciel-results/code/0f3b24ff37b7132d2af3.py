"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Strategy: 
    1. Frame support selection as finding a large set of 4-subsets of {0..10} such that 
       any two subsets intersect in at most 2 elements.
    2. Use fast vectorized conflict matrix computation.
    3. Use randomized greedy MIS with multiple restarts and improved local search.
    4. For each selected support, add all 16 sign variants.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate all 4-element supports as bitmasks
    supports = []
    for comb in combinations(range(D), 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    
    n_sup = len(supports)  # 330
    
    # 2. Fast vectorized conflict matrix computation
    sup_arr = np.array(supports, dtype=np.int64)
    conflict = np.zeros((n_sup, n_sup), dtype=bool)
    
    bit_count_lut = np.array([i.bit_count() for i in range(2048)], dtype=np.uint8)
    
    chunk_size = 50
    for i_start in range(0, n_sup, chunk_size):
        i_end = min(i_start + chunk_size, n_sup)
        pairwise_and = sup_arr[i_start:i_end, None] & sup_arr[None, :]
        pairwise_idx = pairwise_and.astype(np.intp)
        inter_counts = bit_count_lut[pairwise_idx]
        conflict[i_start:i_end, :] = inter_counts > 2
    
    # 3. Fast greedy MIS with randomized tie-breaking
    def _fast_greedy_mis(conflict_matrix, rng):
        n = len(conflict_matrix)
        selected = np.zeros(n, dtype=bool)
        available = np.ones(n, dtype=bool)
        
        cur_deg = conflict_matrix.sum(axis=1).astype(np.int32)
        
        while available.any():
            avail_idx = np.where(available)[0]
            if len(avail_idx) == 0:
                break
            
            min_deg = cur_deg[avail_idx].min()
            min_mask = (cur_deg == min_deg) & available
            candidates = np.where(min_mask)[0]
            
            v = candidates[rng.integers(0, len(candidates))]
            selected[v] = True
            available[v] = False
            
            neighbors = np.where(conflict_matrix[v] & available)[0]
            if len(neighbors) > 0:
                available[neighbors] = False
                avail_after = np.where(available)[0]
                if len(avail_after) > 0:
                    cur_deg[avail_after] -= conflict_matrix[avail_after, v]
        
        return selected
    
    def _local_search_add(selected, conflict_matrix, rng, max_iter=500):
        """
        Improved local search with 2-swap moves and better exploration.
        """
        n = len(selected)
        current_sel = selected.copy()
        current_size = current_sel.sum()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            unselected = np.where(~current_sel)[0]
            if len(unselected) == 0:
                break
            
            # Count conflicts with selected nodes
            conf_with_sel = conflict_matrix[unselected, :].dot(current_sel.astype(np.int32))
            
            # Try to add nodes with 0 conflicts
            can_add = np.where(conf_with_sel == 0)[0]
            if len(can_add) > 0:
                u = unselected[can_add[rng.integers(0, len(can_add))]]
                current_sel[u] = True
                current_size += 1
                continue
            
            # Try 1-swap: replace one selected node with one unselected
            conf_1 = np.where(conf_with_sel == 1)[0]
            if len(conf_1) > 0:
                # Try all 1-swaps, pick the one that minimizes new conflicts
                best_swap = None
                best_new_conf = float('inf')
                for u_idx in conf_1:
                    u = unselected[u_idx]
                    conflicting_sel = np.where(conflict_matrix[u] & current_sel)[0]
                    if len(conflicting_sel) == 1:
                        v = conflicting_sel[0]
                        # After removing v and adding u, count new conflicts for u
                        # u conflicts with 1 selected node (v), so after swap it has 0 conflicts with remaining selected
                        # But we need to check if removing v allows other unselected nodes to be added later
                        # For now, just do the swap
                        current_sel[v] = False
                        current_sel[u] = True
                        best_swap = (u, v)
                        break
                
                if best_swap is None:
                    # Try random 1-swap
                    u_idx = conf_1[rng.integers(0, len(conf_1))]
                    u = unselected[u_idx]
                    conflicting_sel = np.where(conflict_matrix[u] & current_sel)[0]
                    if len(conflicting_sel) == 1:
                        v = conflicting_sel[0]
                        current_sel[v] = False
                        current_sel[u] = True
                        continue
            
            # Try 2-swap: remove 2 selected, add 2 unselected
            # Find unselected nodes with 2 conflicts
            conf_2 = np.where(conf_with_sel == 2)[0]
            if len(conf_2) >= 2:
                # Try random pairs of 2-conflict nodes
                for _ in range(10):  # Limited attempts
                    u1_idx, u2_idx = rng.choice(len(conf_2), size=2, replace=False)
                    u1 = unselected[conf_2[u1_idx]]
                    u2 = unselected[conf_2[u2_idx]]
                    
                    # Get their conflicting selected nodes
                    conf1 = np.where(conflict_matrix[u1] & current_sel)[0]
                    conf2 = np.where(conflict_matrix[u2] & current_sel)[0]
                    
                    # Check if we can remove 2 selected nodes and add both u1 and u2
                    # The union of conflicts must be exactly 2 nodes
                    all_conf = np.unique(np.concatenate([conf1, conf2]))
                    if len(all_conf) == 2:
                        v1, v2 = all_conf
                        # Check that u1 and u2 don't conflict with each other
                        if not conflict_matrix[u1, u2]:
                            current_sel[v1] = False
                            current_sel[v2] = False
                            current_sel[u1] = True
                            current_sel[u2] = True
                            break
            
            # Random perturbation to escape local minima
            if rng.random() < 0.1:
                sel_idx = np.where(current_sel)[0]
                if len(sel_idx) > 0:
                    v = sel_idx[rng.integers(0, len(sel_idx))]
                    current_sel[v] = False
                    current_size -= 1
        
        return current_sel
    
    best_sel = None
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Run multiple greedy constructions with local search
    num_runs = 200
    for run in range(num_runs):
        if time.time() > deadline - 10:
            break
        
        # Greedy construction
        sel = _fast_greedy_mis(conflict, rng)
        size = sel.sum()
        
        # Local search to try to improve
        if size > best_size or (size == best_size and run % 3 == 0):
            sel_improved = _local_search_add(sel, conflict, rng, max_iter=500)
            size_improved = sel_improved.sum()
            
            if size_improved > best_size:
                best_size = size_improved
                best_sel = sel_improved.copy()
            elif size > best_size:
                best_size = size
                best_sel = sel.copy()
        elif size > best_size:
            best_size = size
            best_sel = sel.copy()
    
    # 4. Construct final vectors with all 16 sign variants per support
    vectors = []
    for idx in range(n_sup):
        if best_sel[idx]:
            mask = supports[idx]
            inds = [i for i in range(D) if (mask >> i) & 1]
            for sign_mask in range(16):
                v = np.zeros(D, dtype=np.int64)
                for j, idx_val in enumerate(inds):
                    if sign_mask & (1 << j):
                        v[idx_val] = 1
                    else:
                        v[idx_val] = -1
                vectors.append(v)
                
    if not vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = np.array(vectors, dtype=np.int64)
    
    # Final safety check
    n_vec = len(result)
    if n_vec > 1:
        norms_sq = np.sum(result ** 2, axis=1)
        max_norm_sq = norms_sq.max()
        dots = result @ result.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, 0)
        min_dist_sq = dist_sq.min()
        
        if min_dist_sq < max_norm_sq:
            # Fallback: remove violating vectors
            keep = np.ones(n_vec, dtype=bool)
            for i in range(n_vec):
                for j in range(i + 1, n_vec):
                    if dist_sq[i, j] < max_norm_sq:
                        keep[j] = False
                        break
            result = result[keep]
    
    return result

# EVOLVE-BLOCK-END
