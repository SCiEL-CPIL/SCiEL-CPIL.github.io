import itertools
import time

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_t_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # Precompute point masks (19-bit) for each block
    point_masks = [0] * n_blocks
    # Precompute t-subset masks (large bitmasks) for each block
    t_masks = [0] * n_blocks
    
    for i, block in enumerate(k_subsets):
        pm = 0
        for p in block:
            pm |= (1 << p)
        point_masks[i] = pm
        
        tm = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            tm |= (1 << idx)
        t_masks[i] = tm

    # 3. Greedy Set Cover to get initial solution
    uncovered = all_t_mask
    selected_greedy = []
    used_blocks = [False] * n_blocks
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(n_blocks):
            if used_blocks[i]:
                continue
            gain = (t_masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        selected_greedy.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~t_masks[best_idx]
        
    current_solution = selected_greedy[:]

    # 4. Local Search
    # We maintain the solution as a list of block indices.
    # We use two heuristics:
    # a. Remove redundant blocks (if removing A doesn't uncover any t-subset)
    # b. Replace multiple blocks with a single superset block (if A, B subset of C, replace A,B with C)
    
    def get_solution_masks(sol):
        return [t_masks[i] for i in sol]

    def local_search(sol):
        # Convert to list of indices
        sol = sol[:]
        n_sol = len(sol)
        
        # Precompute point masks for current solution blocks
        sol_point_masks = [point_masks[i] for i in sol]
        
        improved = True
        while improved:
            improved = False
            
            # --- Heuristic 1: Remove Redundant Blocks ---
            # Count coverage of each t-subset by current solution
            # We can do this efficiently by summing bit counts? No, we need per-bit counts.
            # n_t_subsets is ~11628.
            # We can use an array of counts.
            
            cover_counts = [0] * n_t_subsets
            for idx in sol:
                mask = t_masks[idx]
                # Iterate set bits
                m = mask
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    cover_counts[bit_idx] += 1
                    m ^= lsb
            
            redundant_found = False
            for i in range(len(sol)):
                idx = sol[i]
                mask = t_masks[idx]
                
                # Tentatively remove
                m = mask
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    cover_counts[bit_idx] -= 1
                    if cover_counts[bit_idx] == 0:
                        # This t-subset would be uncovered
                        # Restore and break
                        cover_counts[bit_idx] += 1
                        # Restore rest
                        m2 = m ^ lsb
                        while m2:
                            lsb2 = m2 & -m2
                            bit_idx2 = lsb2.bit_length() - 1
                            cover_counts[bit_idx2] += 1
                            m2 ^= lsb2
                        redundant_found = True
                        break
                    m ^= lsb
                
                if redundant_found:
                    break
                
                # If we get here, the block was redundant
                sol.pop(i)
                improved = True
                break # Restart loop with new solution
            
            if improved:
                continue
                
            # --- Heuristic 2: Replace with Superset ---
            # Check if any block C in universe contains >= 2 blocks from sol
            # If so, replace them with C.
            
            # To speed up, we can iterate through universe blocks.
            # For each C, check how many sol blocks are subsets of C.
            
            # Optimization: Only check C if it's not already in sol?
            # If C is in sol, it contains itself (1 block). We need >= 2.
            
            replacement_found = False
            for c_idx in range(n_blocks):
                c_pm = point_masks[c_idx]
                
                # Count how many blocks in sol are subsets of c_idx
                subset_indices = []
                for i, idx in enumerate(sol):
                    if (point_masks[idx] & c_pm) == point_masks[idx]:
                        subset_indices.append(i)
                        
                if len(subset_indices) >= 2:
                    # Remove these blocks from sol
                    # We need to remove them carefully
                    # Create a set of indices to remove
                    remove_set = set(subset_indices)
                    new_sol = [sol[i] for i in range(len(sol)) if i not in remove_set]
                    
                    # Add c_idx
                    if c_idx not in new_sol:
                        new_sol.append(c_idx)
                    else:
                        # If c_idx is already in new_sol, we just removed its subsets?
                        # Wait, if c_idx is in sol, and we found >= 2 subsets, one of them might be c_idx itself?
                        # If c_idx is in sol, point_masks[c_idx] is a subset of itself.
                        # So subset_indices will include the index of c_idx in sol.
                        # If len(subset_indices) >= 2, we replace all of them with c_idx.
                        # Since c_idx is one of them, the net change is removing the *other* subsets.
                        pass
                    
                    sol = new_sol
                    improved = True
                    replacement_found = True
                    break # Restart loop
            
            if replacement_found:
                continue
                
            # No improvement
            break
            
        return sol

    final_solution = local_search(current_solution)
    
    return [list(k_subsets[i]) for i in final_solution]
