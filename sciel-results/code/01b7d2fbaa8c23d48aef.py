import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    D = 11
    
    # Generate Weight 4 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Compute dot products
    dots = np.dot(candidates, candidates.T)
    
    # Conflict if dist_sq < 4
    # dist_sq = 4 + 4 - 2*dots = 8 - 2*dots
    # 8 - 2*dots < 4 => 2*dots > 4 => dots > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to Python Integers for bitset operations
    # We need to map index i to bit i
    # neighbors[i] is an integer with bits set for j where conflicts[i,j] is True
    neighbors = [0] * n_candidates
    for i in range(n_candidates):
        # Get indices of True in conflicts[i]
        idxs = np.where(conflicts[i])[0]
        mask = 0
        for j in idxs:
            mask |= (1 << j)
        neighbors[i] = mask
        
    all_mask = (1 << n_candidates) - 1
    
    def popcount(x):
        return bin(x).count("1")
        
    def pick_min_degree(available_mask):
        # Find node in available_mask with minimum degree in available subgraph
        min_deg = float('inf')
        best_v = -1
        # Iterate through set bits
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            m -= lsb
            
            # Degree of v in available subgraph
            deg = popcount(available_mask & neighbors[v])
            if deg < min_deg:
                min_deg = deg
                best_v = v
            if time.time() > deadline - 5.0:
                return -1
        return best_v

    def greedy_fill(available_mask, selected_mask):
        while available_mask:
            v = pick_min_degree(available_mask)
            if v == -1:
                break
            # Add v
            selected_mask |= (1 << v)
            available_mask &= ~(1 << v)
            available_mask &= ~neighbors[v]
            
            if time.time() > deadline - 5.0:
                break
        return selected_mask, available_mask

    def compute_available(selected_mask):
        # Nodes not in selected and not conflicting with any node in selected
        avail = all_mask & ~selected_mask
        # For each u in selected, remove neighbors[u] from avail
        # This is slow if done one by one.
        # But we can just recompute from scratch?
        # Or maintain it?
        # Let's recompute:
        # avail = all_mask
        # avail &= ~selected_mask
        # s = selected_mask
        # while s:
        #   lsb = s & -s
        #   u = lsb.bit_length() - 1
        #   s -= lsb
        #   avail &= ~neighbors[u]
        # return avail
        avail = all_mask & ~selected_mask
        s = selected_mask
        while s:
            lsb = s & -s
            u = lsb.bit_length() - 1
            s -= lsb
            avail &= ~neighbors[u]
        return avail

    best_mask = 0
    best_size = 0
    
    # Run multiple trials
    num_trials = 10
    for trial in range(num_trials):
        if time.time() > deadline - 10.0:
            break
            
        # Randomize order or weights?
        # Just run greedy with random tie-breaking?
        # pick_min_degree is deterministic.
        # Let's add a random perturbation to the degree calculation?
        # Or just start with a random subset?
        
        # Strategy: Start with empty, greedy fill.
        sel, avail = greedy_fill(all_mask, 0)
        
        size = popcount(sel)
        if size > best_size:
            best_size = size
            best_mask = sel
            
        # Local Search
        if time.time() < deadline - 5.0:
            current_sel = sel
            current_avail = avail
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                
                # 1. Greedy Add
                while current_avail:
                    v = pick_min_degree(current_avail)
                    if v == -1:
                        break
                    current_sel |= (1 << v)
                    current_avail &= ~(1 << v)
                    current_avail &= ~neighbors[v]
                    improved = True
                    
                size = popcount(current_sel)
                if size > best_size:
                    best_size = size
                    best_mask = current_sel
                    
                # 2. Try Swaps
                # Pick a random node from current_sel to remove
                if current_sel:
                    # Get list of nodes in current_sel
                    s = current_sel
                    nodes = []
                    while s:
                        lsb = s & -s
                        u = lsb.bit_length() - 1
                        s -= lsb
                        nodes.append(u)
                    
                    # Sample up to 10 random nodes
                    if len(nodes) > 10:
                        sample = random.sample(nodes, 10)
                    else:
                        sample = nodes
                        
                    for u in sample:
                        if time.time() > deadline - 5.0:
                            break
                            
                        # Remove u
                        new_sel = current_sel & ~(1 << u)
                        # Compute new available
                        # Nodes that were blocked by u and are not in new_sel
                        # blocked_by_u = neighbors[u] & ~new_sel
                        # new_avail = current_avail | blocked_by_u | (1 << u)
                        # But we must also remove neighbors of new_sel?
                        # No, current_avail is already "clean" w.r.t current_sel.
                        # When we remove u, some nodes in neighbors[u] become available.
                        # u itself becomes available.
                        new_avail = current_avail | (neighbors[u] & ~new_sel) | (1 << u)
                        
                        # Greedy add from new_avail
                        temp_sel = new_sel
                        temp_avail = new_avail
                        while temp_avail:
                            v = pick_min_degree(temp_avail)
                            if v == -1:
                                break
                            temp_sel |= (1 << v)
                            temp_avail &= ~(1 << v)
                            temp_avail &= ~neighbors[v]
                            
                        temp_size = popcount(temp_sel)
                        if temp_size > size:
                            current_sel = temp_sel
                            current_avail = temp_avail
                            size = temp_size
                            improved = True
                            if size > best_size:
                                best_size = size
                                best_mask = current_sel
                            break # Restart loop with new state

    # Convert best_mask back to array of indices
    selected_indices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        m -= lsb
        selected_indices.append(v)
        
    if not selected_indices:
        # Fallback
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
        
    return candidates[selected_indices]
