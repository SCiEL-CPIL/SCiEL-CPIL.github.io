import itertools
import time

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * len(k_subsets)
    block_t_indices = [None] * len(k_subsets)
    # Precompute for each t-subset, the list of blocks that cover it
    # This is useful for local search
    # Memory: 11628 lists. Total entries = 75000 * 56 = 4.2M.
    # 4.2M integers is ~35MB. Acceptable.
    t_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for i, block in enumerate(k_subsets):
        mask = 0
        idxs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
            idxs.append(idx)
            t_to_blocks[idx].append(i)
        masks[i] = mask
        block_t_indices[i] = idxs

    # 4. Greedy Set Cover
    uncovered = all_mask
    selected_greedy = []
    used_blocks = [False] * len(k_subsets)
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(len(k_subsets)):
            if used_blocks[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        selected_greedy.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    current_best = len(selected_greedy)
    best_solution = selected_greedy[:]

    # 5. Efficient Redundancy Removal using Cover Counts
    def get_cover_counts(blocks_list):
        counts = [0] * n_t_subsets
        for b_idx in blocks_list:
            for t_idx in block_t_indices[b_idx]:
                counts[t_idx] += 1
        return counts

    def remove_redundant(blocks_list):
        if not blocks_list:
            return []
        cover_count = get_cover_counts(blocks_list)
        new_blocks = []
        for b_idx in blocks_list:
            redundant = True
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
            else:
                new_blocks.append(b_idx)
        return new_blocks

    best_solution = remove_redundant(best_solution)
    current_best = len(best_solution)

    # 6. Local Search: 1-Exchange with Redundancy Check
    # Try to swap a block A with B if B covers A's unique parts, and then we can remove another block.
    
    in_solution = [False] * len(k_subsets)
    for b in best_solution:
        in_solution[b] = True
        
    improved = True
    while improved:
        improved = False
        cover_count = get_cover_counts(best_solution)
        
        # Find blocks with unique coverage
        for i, b_idx in enumerate(best_solution):
            unique_ts = []
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] == 1:
                    unique_ts.append(t_idx)
            
            if not unique_ts:
                # Redundant, should have been removed, but just in case
                # Remove it
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                best_solution.pop(i)
                in_solution[b_idx] = False
                improved = True
                break # Restart loop as indices changed
                
            if len(unique_ts) == 1:
                t0 = unique_ts[0]
                # Find a block B not in solution that covers t0
                # To minimize changes, we might prefer B that covers few other things?
                # Or just any B.
                for b2 in t_to_blocks[t0]:
                    if in_solution[b2]:
                        continue
                    # Try swap b_idx with b2
                    # Check if b2 covers all unique_ts (it covers t0)
                    # What about other t's in b_idx?
                    # If we swap, the unique_ts of b_idx are covered by b2.
                    # Other t's in b_idx are covered by others (count > 1) or become unique?
                    # Wait, if we remove b_idx, the t's in b_idx that had count 2 will have count 1.
                    # If b2 covers them, their count stays >= 1.
                    # We need to ensure the solution remains valid.
                    # Validity: All t's must have count >= 1.
                    
                    # Let's simulate the swap
                    # Remove b_idx:
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    # Add b2:
                    for t_idx in block_t_indices[b2]:
                        cover_count[t_idx] += 1
                        
                    # Check if any t has count 0
                    valid = True
                    for t_idx in block_t_indices[b_idx]:
                        if cover_count[t_idx] == 0:
                            valid = False
                            break
                    if not valid:
                        # Revert
                        for t_idx in block_t_indices[b2]:
                            cover_count[t_idx] -= 1
                        for t_idx in block_t_indices[b_idx]:
                            cover_count[t_idx] += 1
                        continue
                        
                    # If valid, perform swap
                    in_solution[b_idx] = False
                    in_solution[b2] = True
                    best_solution[i] = b2
                    
                    # Now try to remove redundant blocks
                    # We can just run remove_redundant on the new solution
                    # But that is expensive.
                    # Let's just mark improved and break to restart the outer loop
                    improved = True
                    break
                    
                if improved:
                    break
            # If len(unique_ts) > 1, skip for now (complex)
            
    return [list(k_subsets[i]) for i in best_solution]
