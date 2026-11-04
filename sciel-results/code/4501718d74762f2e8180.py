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
    3. Use randomized greedy MIS with multiple restarts and local search.
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
    # Convert supports to numpy array for vectorized bit operations
    sup_arr = np.array(supports, dtype=np.int64)
    
    # Compute pairwise AND using broadcasting: shape (n_sup, n_sup)
    # But this is 330x330 = 108900 entries, each an int64 AND operation
    # We can do this in chunks to manage memory
    conflict = np.zeros((n_sup, n_sup), dtype=bool)
    
    # Vectorized approach: for each pair (i,j), compute bit_count of (sup_i & sup_j)
    # Use numpy's ability to do element-wise ops
    # sup_arr[:, None] & sup_arr[None, :] gives shape (330, 330)
    # Then we need bit_count on each element
    
    # NumPy doesn't have bit_count directly, so we use a lookup or manual count
    # For 11-bit integers, we can precompute a lookup table for 0..2047
    bit_count_lut = np.array([i.bit_count() for i in range(2048)], dtype=np.uint8)
    
    # Compute all pairwise ANDs
    # To save memory, do it in row chunks
    chunk_size = 50
    for i_start in range(0, n_sup, chunk_size):
        i_end = min(i_start + chunk_size, n_sup)
        # Shape: (chunk, n_sup)
        pairwise_and = sup_arr[i_start:i_end, None] & sup_arr[None, :]
        # Convert to indices for LUT lookup
        pairwise_idx = pairwise_and.astype(np.intp)
        inter_counts = bit_count_lut[pairwise_idx]
        conflict[i_start:i_end, :] = inter_counts > 2
    
    # Make symmetric (though it should already be)
    # conflict = conflict | conflict.T  # Just in case
    
    # 3. Fast greedy MIS with randomized tie-breaking
    def _fast_greedy_mis(conflict_matrix, rng):
        n = len(conflict_matrix)
        selected = np.zeros(n, dtype=bool)
        available = np.ones(n, dtype=bool)
        
        # Current degree: number of available neighbors for each node
        # Initially, degree[i] = conflict_matrix[i].sum()
        cur_deg = conflict_matrix.sum(axis=1).astype(np.int32)
        
        while available.any():
            # Get indices of available nodes
            avail_idx = np.where(available)[0]
            if len(avail_idx) == 0:
                break
            
            # Find min degree among available nodes
            # cur_deg[avail_idx] gives degrees for available nodes
            min_deg = cur_deg[avail_idx].min()
            
            # Find all nodes with this min degree
            min_mask = (cur_deg == min_deg) & available
            candidates = np.where(min_mask)[0]
            
            # Randomly pick from candidates
            v = candidates[rng.integers(0, len(candidates))]
            selected[v] = True
            available[v] = False
            
            # Update: remove v's neighbors from available
            neighbors = np.where(conflict_matrix[v] & available)[0]
            if len(neighbors) > 0:
                available[neighbors] = False
                # Update degrees: for each remaining available node, 
                # decrement if it was connected to v
                # cur_deg[available] -= conflict_matrix[available, v]
                # But we need to be careful: only decrement for nodes that are still available
                # Actually, we can just recompute or use a simpler update
                # For speed, let's just recompute degrees from scratch occasionally
                # Or: cur_deg[available] -= conflict_matrix[np.where(available)[0], v]
                avail_after = np.where(available)[0]
                if len(avail_after) > 0:
                    cur_deg[avail_after] -= conflict_matrix[avail_after, v]
        
        return selected
    
    def _local_search_add(selected, conflict_matrix, rng, max_iter=500):
        """
        Try to add more nodes by removing conflicting nodes.
        For each unselected node u, check how many selected nodes it conflicts with.
        If it conflicts with k selected nodes, we can remove those k and add u.
        This is profitable if k < 1 (impossible) or if removing k and adding 1 is neutral,
        but might allow further additions.
        
        Better: try 1-remove, 1-add swaps that maintain size but improve "quality"
        (lower total degree), which might enable future additions.
        
        Or: try to find a node u that conflicts with exactly 1 selected node v.
        Remove v, add u. Same size, but u might have lower degree.
        """
        n = len(selected)
        current_sel = selected.copy()
        current_size = current_sel.sum()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Find unselected nodes and their conflict count with selected nodes
            unselected = np.where(~current_sel)[0]
            if len(unselected) == 0:
                break
            
            # For each unselected node, count how many selected nodes it conflicts with
            # conflict_matrix[unselected, :].sum(axis=1) gives conflicts with ALL nodes
            # We want conflicts with only selected nodes
            conf_with_sel = conflict_matrix[unselected, :].dot(current_sel.astype(np.int32))
            
            # Find nodes that conflict with exactly 0 selected nodes (can be added directly)
            # This shouldn't happen if we started from a maximal IS, but check anyway
            can_add = np.where(conf_with_sel == 0)[0]
            if len(can_add) > 0:
                # Add one of them
                u = unselected[can_add[rng.integers(0, len(can_add))]]
                current_sel[u] = True
                current_size += 1
                continue
            
            # Find nodes that conflict with exactly 1 selected node
            conf_1 = np.where(conf_with_sel == 1)[0]
            if len(conf_1) > 0:
                # Pick one randomly
                u = unselected[conf_1[rng.integers(0, len(conf_1))]]
                # Find which selected node it conflicts with
                conflicting_sel = np.where(conflict_matrix[u] & current_sel)[0]
                v = conflicting_sel[0]
                
                # Swap: remove v, add u
                current_sel[v] = False
                current_sel[u] = True
                # Size stays the same, but this might enable future additions
                continue
            
            # No direct improvements, try random swap to escape local minima
            # Pick a random selected node and a random unselected node, swap if possible
            sel_idx = np.where(current_sel)[0]
            if len(sel_idx) == 0 or len(unselected) == 0:
                break
            
            v = sel_idx[rng.integers(0, len(sel_idx))]
            u = unselected[rng.integers(0, len(unselected))]
            
            # Check if u can replace v: u must not conflict with any selected node except v
            conf_with_others = conflict_matrix[u, :].copy()
            conf_with_others[v] = False  # Ignore conflict with v
            if not (conf_with_others & current_sel).any():
                current_sel[v] = False
                current_sel[u] = True
        
        return current_sel
    
    best_sel = None
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Run multiple greedy constructions with local search
    num_runs = 100
    for run in range(num_runs):
        if time.time() > deadline - 15:
            break
        
        # Greedy construction
        sel = _fast_greedy_mis(conflict, rng)
        size = sel.sum()
        
        # Local search to try to improve
        if size > best_size or (size == best_size and run % 5 == 0):
            sel_improved = _local_search_add(sel, conflict, rng, max_iter=300)
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
            # This should not happen, but handle gracefully
            keep = np.ones(n_vec, dtype=bool)
            for i in range(n_vec):
                for j in range(i + 1, n_vec):
                    if dist_sq[i, j] < max_norm_sq:
                        # Remove the one with larger index (arbitrary)
                        keep[j] = False
                        break
            result = result[keep]
    
    return result

# EVOLVE-BLOCK-END
