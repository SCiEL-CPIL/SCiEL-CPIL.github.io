# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 supports
    supports = list(itertools.combinations(range(11), 4))
    n_supports = len(supports) # 330
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Build conflict masks
    # conflict_mask[i] is an int with bit j set if support i conflicts with support j
    # Conflict if intersection size >= 3
    conflict_mask = [0] * n_supports
    
    for i in range(n_supports):
        si = supports[i]
        for j in range(i + 1, n_supports):
            sj = supports[j]
            # Check intersection size
            # Since both are tuples of 4, we can check quickly
            # Or use set intersection
            if len(set(si) & set(sj)) >= 3:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    # Precompute neighbor lists for faster iteration if needed, but bitmasks are good
    # All supports mask
    all_mask = (1 << n_supports) - 1
    
    rng = np.random.default_rng(42)
    
    def greedy_independent_set():
        current_set = []
        available = all_mask
        
        # Heuristic: pick node with min degree in available subgraph
        # Degree in available subgraph for node i is popcount(conflict_mask[i] & available)
        # This is expensive to compute for all.
        # Simple greedy: pick random from available, or min index.
        # Let's try min-degree greedy.
        
        while available:
            # Find node in available with min conflicts within available
            best_node = -1
            min_deg = 10000
            # Iterate through set bits in available
            avail_list = []
            temp = available
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                avail_list.append(idx)
                temp ^= lsb
                
            for idx in avail_list:
                deg = bin(conflict_mask[idx] & available).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
                    
            if best_node == -1:
                break
                
            current_set.append(best_node)
            available &= ~(conflict_mask[best_node] | (1 << best_node))
            
        return current_set

    def sa_local_search(init_set, max_time=10.0):
        current_set = list(init_set)
        current_mask = 0
        for idx in current_set:
            current_mask |= (1 << idx)
            
        # Compute available
        available = all_mask & ~current_mask
        # Remove conflicts from available
        for idx in current_set:
            available &= ~conflict_mask[idx]
            
        best_set = list(current_set)
        best_size = len(current_set)
        
        T = 5.0
        cooling = 0.999
        min_T = 0.1
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            move_type = rng.random()
            
            if move_type < 0.5:
                # Add move
                if available:
                    # Pick a node from available
                    # Prefer nodes with fewer conflicts (lower degree in current graph? No, just random or heuristic)
                    avail_list = []
                    temp = available
                    while temp:
                        lsb = temp & (-temp)
                        idx = lsb.bit_length() - 1
                        avail_list.append(idx)
                        temp ^= lsb
                        
                    if avail_list:
                        # Heuristic: pick one that conflicts with fewest in current_set?
                        # Actually, all in available have 0 conflicts with current_set.
                        # So just pick random.
                        idx = avail_list[rng.integers(len(avail_list))]
                        
                        # Add idx
                        current_set.append(idx)
                        current_mask |= (1 << idx)
                        
                        # Update available: remove idx and its neighbors
                        available &= ~(conflict_mask[idx] | (1 << idx))
                        
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                            
            elif move_type < 0.8:
                # Remove move
                if current_set:
                    idx = current_set.pop(rng.integers(len(current_set)))
                    current_mask &= ~(1 << idx)
                    
                    # Update available: add back idx and its neighbors that are not in current_set
                    # Neighbors of idx are in conflict_mask[idx]
                    # We need to add (conflict_mask[idx] | (1<<idx)) & ~current_mask to available
                    available |= (conflict_mask[idx] | (1 << idx)) & ~current_mask
                    
            else:
                # Swap move: remove one, add one
                if current_set and available:
                    # Remove random idx
                    idx_remove = current_set.pop(rng.integers(len(current_set)))
                    current_mask &= ~(1 << idx_remove)
                    
                    # Temporarily update available to include idx_remove's neighbors
                    temp_available = available | ((conflict_mask[idx_remove] | (1 << idx_remove)) & ~current_mask)
                    
                    if temp_available:
                        # Pick a node to add from temp_available
                        # It must not conflict with the rest of current_set
                        # Since we removed idx_remove, its neighbors are now potentially available
                        # But we must ensure the new node doesn't conflict with other nodes in current_set
                        # The `available` mask before removal already excluded nodes conflicting with current_set.
                        # When we remove idx_remove, we add back its neighbors.
                        # Do these neighbors conflict with other nodes in current_set?
                        # Yes, they might.
                        # So we must filter temp_available against current_set.
                        
                        valid_adds = temp_available
                        for idx_cur in current_set:
                            valid_adds &= ~conflict_mask[idx_cur]
                            
                        if valid_adds:
                            # Pick one
                            avail_list = []
                            temp = valid_adds
                            while temp:
                                lsb = temp & (-temp)
                                idx = lsb.bit_length() - 1
                                avail_list.append(idx)
                                temp ^= lsb
                                
                            if avail_list:
                                idx_add = avail_list[rng.integers(len(avail_list))]
                                current_set.append(idx_add)
                                current_mask |= (1 << idx_add)
                                
                                # Update available properly
                                # Remove idx_add and its neighbors from available
                                # Note: available currently is the OLD available (before removal) 
                                # We need to recalculate or update carefully.
                                # Easier: Recalculate available from scratch for swap? 
                                # Or update:
                                # available was updated to include idx_remove's neighbors.
                                # Now we add idx_add.
                                # available = (available | (conflict_mask[idx_remove] | (1<<idx_remove))) & ~current_mask
                                # Then remove idx_add's neighbors.
                                
                                # Let's just recalculate available for simplicity in SA step if it's fast enough
                                # 330 bits is very fast.
                                available = all_mask & ~current_mask
                                for idx_cur in current_set:
                                    available &= ~conflict_mask[idx_cur]
                                    
                                if len(current_set) > best_size:
                                    best_size = len(current_set)
                                    best_set = list(current_set)
                            else:
                                # Revert removal
                                current_set.append(idx_remove)
                                current_mask |= (1 << idx_remove)
                                available = all_mask & ~current_mask
                                for idx_cur in current_set:
                                    available &= ~conflict_mask[idx_cur]
                        else:
                            # Revert removal
                            current_set.append(idx_remove)
                            current_mask |= (1 << idx_remove)
                            available = all_mask & ~current_mask
                            for idx_cur in current_set:
                                available &= ~conflict_mask[idx_cur]
                    else:
                        # Revert removal
                        current_set.append(idx_remove)
                        current_mask |= (1 << idx_remove)
                        available = all_mask & ~current_mask
                        for idx_cur in current_set:
                            available &= ~conflict_mask[idx_cur]
                        
            T *= cooling
            if T < min_T:
                T = min_T
                
        return best_set

    best_set = []
    best_size = 0
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Greedy init
        init_set = greedy_independent_set()
        
        # SA
        remaining = deadline - 5 - time.time()
        if remaining < 1:
            break
        
        sa_time = min(remaining * 0.5, 2.0) # Short SA per restart
        sa_result = sa_local_search(init_set, max_time=sa_time)
        
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = sa_result
    
    # Final fill (greedy add)
    current_mask = 0
    for idx in best_set:
        current_mask |= (1 << idx)
        
    available = all_mask & ~current_mask
    for idx in best_set:
        available &= ~conflict_mask[idx]
        
    while available:
        # Pick any
        lsb = available & (-available)
        idx = lsb.bit_length() - 1
        best_set.append(idx)
        available &= ~(conflict_mask[idx] | (1 << idx))
        
    # Build vectors
    final_vectors = []
    
    # Weight 1 vectors
    for pos in range(11):
        for sign in [1, -1]:
            v = [0] * 11
            v[pos] = 2 * sign
            final_vectors.append(v)
            
    # Weight 4 vectors
    for idx in best_set:
        support = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * 11
            for pos, s in zip(support, signs):
                v[pos] = s
            final_vectors.append(v)
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
