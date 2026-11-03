"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pools
    
    # Type A: +/- 2e_i (22 vectors)
    type_a_list = []
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign * 2
            type_a_list.append(v)
    type_a = np.array(type_a_list, dtype=np.int8)
    
    # Type B: +/- e_i +/- e_j +/- e_k +/- e_l (5280 vectors)
    type_b_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_b_list.append(v)
    type_b = np.array(type_b_list, dtype=np.int8)
    n_b = len(type_b)
    
    # 2. Precompute Conflict Matrix for Type B
    # Conflict if dot > 2
    c_b = type_b.astype(np.int32)
    dot_b = c_b @ c_b.T
    conflict_bool = (dot_b > 2).astype(np.uint8)
    np.fill_diagonal(conflict_bool, 0)
    
    # Represent conflicts as a list of integer bitsets for fast subset operations
    # Each integer represents a set of conflicting nodes.
    # n_b = 5280. We need ceil(5280/64) = 83 words per bitset.
    words_per_set = (n_b + 63) // 64
    conflict_bits = [0] * n_b
    for i in range(n_b):
        mask = conflict_bool[i]
        # Convert mask to a list of 64-bit integers
        bits = 0
        for w in range(words_per_set):
            start_idx = w * 64
            end_idx = min(start_idx + 64, n_b)
            if start_idx >= n_b:
                break
            word_val = 0
            for j in range(start_idx, end_idx):
                if mask[j]:
                    word_val |= (1 << (j - start_idx))
            bits |= (word_val << (w * 64))
        conflict_bits[i] = bits

    best_count_b = 0
    best_selected_b = np.zeros(n_b, dtype=np.uint8)
    
    rng = np.random.default_rng(42)
    
    def bitset_popcount(x):
        return bin(x).count('1')
        
    def bitset_and_not(a, b):
        return a & ~b

    def bitset_or(a, b):
        return a | b

    def bitset_subset_check(a, b):
        # Check if a is a subset of b (a & ~b == 0)
        return (a & ~b) == 0

    def greedy_construct_b():
        """
        Greedy selection using bitsets.
        """
        selected_bits = 0
        available_bits = (1 << n_b) - 1 # All nodes available initially
        
        # We need to find the node with minimum degree among available nodes.
        # To do this efficiently, we can maintain a priority queue or just scan.
        # Scanning 5280 nodes is fast.
        
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Find available nodes
            # Instead of extracting all indices, we can iterate through available bits
            # But finding min degree requires checking conflicts with available set.
            # Degree of i in available = popcount(conflict_bits[i] & available_bits)
            
            # Optimization: If we just pick a random available node, it's faster but might be worse.
            # Let's try random pick first for speed, then local search improves it.
            
            # Get list of available indices
            # This conversion is O(n).
            avail_indices = []
            temp = available_bits
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                avail_indices.append(idx)
                temp ^= lsb
            
            if not avail_indices:
                break
                
            # Pick a random available node
            pick_idx = rng.choice(avail_indices)
            
            selected_bits |= (1 << pick_idx)
            # Remove pick and its conflicts from available
            available_bits &= ~(1 << pick_idx)
            available_bits &= ~conflict_bits[pick_idx]
            
        # Convert selected_bits back to array
        selected_arr = np.zeros(n_b, dtype=np.uint8)
        temp = selected_bits
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            selected_arr[idx] = 1
            temp ^= lsb
            
        return selected_arr

    def local_search_b(selected):
        """
        Refine using 1-for-2 moves with bitsets for speed.
        """
        selected_indices = np.where(selected == 1)[0]
        if len(selected_indices) == 0:
            return selected, 0
            
        improved = True
        max_moves = 50
        move_count = 0
        moves_made = 0
        
        # Precompute selected bits
        selected_bits = 0
        for idx in selected_indices:
            selected_bits |= (1 << idx)
            
        # Available bits = ~(selected | conflicts_of_selected)
        # conflicts_of_selected is OR of conflict_bits[i] for i in selected
        conf_sel_bits = 0
        for idx in selected_indices:
            conf_sel_bits |= conflict_bits[idx]
        
        available_bits = ((1 << n_b) - 1) & ~selected_bits & ~conf_sel_bits

        while improved and move_count < max_moves and time.time() < deadline - 2.0:
            improved = False
            move_count += 1
            
            # Try 1-for-2 moves
            # Pick a random selected node v to remove
            v_idx_in_S = rng.integers(len(selected_indices))
            v = selected_indices[v_idx_in_S]
            
            # New available if we remove v
            # available_new = available_bits | conflict_bits[v] | (1 << v)
            # But we need to check if we can add 2 new nodes u, w that are compatible with each other
            # and compatible with (S \ {v}).
            
            # Nodes compatible with S \ {v} are:
            # 1. Currently available nodes (compatible with all S)
            # 2. Nodes that conflict ONLY with v (in conflict_bits[v] but not in conf_of_S_minus_v)
            
            # Conflicts of S minus v:
            conf_S_minus_v = conf_sel_bits & ~conflict_bits[v] # This is not quite right, need to recompute or use inclusion-exclusion?
            # Actually, a node x is compatible with S\{v} if it doesn't conflict with any node in S except possibly v.
            # i.e., conflict_bits[x] & (selected_bits ^ (1<<v)) == 0
            
            # This check is expensive. Let's stick to the previous array-based local search but optimized?
            # The bitset overhead for complex local search logic might be high.
            
            # Alternative: Use the array-based local search but run it more times.
            # The previous code ran 100 restarts.
            # Let's just run the array-based local search but with a tighter loop and more restarts if time permits.
            
            # For now, return the greedy result without local search if bitset local search is too complex to implement correctly/safely in this scratchpad.
            # I will fall back to the array-based local search from the previous code but ensure it is distinct.
            
            # Let's implement a simple 1-for-1 swap or 1-for-2 swap using arrays but optimized.
            pass
        
        return selected, moves_made

    # Fallback: Use the array-based approach from the previous code but with modifications to make it distinct.
    # We will re-implement the array-based local search here to ensure validity.
    
    def local_search_array(selected):
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        improved = True
        max_moves = 20 # Keep low to allow more restarts
        move_count = 0
        moves_made = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            conflict_with_S = conflict_bool[:, sel_indices]
            total_inc = conflict_with_S.sum(axis=1)
            
            base_mask = (total_inc == 0) & (selected == 0)
            
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v_idx_in_S = idx
                v = sel_indices[v_idx_in_S]
                
                cond_mask = (total_inc == 1) & (conflict_with_S[:, v_idx_in_S] == 1) & (selected == 0)
                Cand_v = base_mask | cond_mask
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                sub_conflict = conflict_bool[np.ix_(cand_indices, cand_indices)]
                # Check if any pair is compatible
                # sub_conflict[i][j] == 0 means compatible
                # We need to find i != j such that sub_conflict[i][j] == 0
                # Note: sub_conflict is symmetric.
                
                # Find a compatible pair
                found = False
                for i in range(len(cand_indices)):
                    if found: break
                    # Check if cand_indices[i] is compatible with any other candidate
                    # Mask of compatible nodes with i among candidates
                    compat_mask = (sub_conflict[i] == 0)
                    # Exclude i itself
                    compat_mask[i] = False
                    if np.any(compat_mask):
                        j_idx = np.where(compat_mask)[0][0]
                        u = cand_indices[i]
                        w = cand_indices[j_idx]
                        
                        selected[v] = 0
                        selected[u] = 1
                        selected[w] = 1
                        improved = True
                        moves_made += 1
                        found = True
                        break
                        
        return selected, moves_made

    # Run multiple restarts
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        selected = greedy_construct_b()
        current_count = selected.sum()
        
        selected, moves = local_search_array(selected)
        current_count = selected.sum()
        
        if current_count > best_count_b:
            best_count_b = current_count
            best_selected_b = selected.copy()

    # 4. Combine Results
    if best_count_b == 0:
        return type_a
    
    selected_type_b = type_b[best_selected_b == 1]
    result = np.vstack([type_a, selected_type_b])
    
    return result.astype(np.int64)

# EVOLVE-BLOCK-END
