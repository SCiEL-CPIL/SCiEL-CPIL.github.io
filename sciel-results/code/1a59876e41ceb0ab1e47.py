import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    # Use a safety margin to avoid timeout
    deadline = start_time + 145.0 
    
    # 1. Precompute all t-subsets and assign IDs
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Precompute all k-subsets (blocks) and their masks
    # A mask is an integer where bit i is 1 if the block covers t-subset i
    k_subsets = list(itertools.combinations(range(v), k))
    block_masks = []
    block_lists = []
    
    for block in k_subsets:
        mask = 0
        # For each t-subset within this block, set the corresponding bit
        for ts in itertools.combinations(block, t):
            mask |= (1 << t_subset_to_id[ts])
        block_masks.append(mask)
        block_lists.append(list(block))
        
    num_blocks = len(k_subsets)
    
    # 3. Greedy Set Cover
    uncovered = (1 << num_t_subsets) - 1
    selected_indices = []
    
    # We iterate until uncovered is 0
    while uncovered:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_gain = -1
        
        # Find the block that covers the most uncovered subsets
        for i in range(num_blocks):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            break # Should not happen if uncovered is non-zero and blocks exist
            
        selected_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # 4. Reduction Phase
    # Maintain cover counts for each t-subset
    cover_count = [0] * num_t_subsets
    for idx in selected_indices:
        block = block_lists[idx]
        for ts in itertools.combinations(block, t):
            cover_count[t_subset_to_id[ts]] += 1
            
    # Try to remove redundant blocks
    # Iterate in reverse order (greedy picks 'tail' blocks last, which are often redundant)
    new_selected_indices = []
    for idx in reversed(selected_indices):
        if time.time() > deadline:
            new_selected_indices.append(idx)
            continue
            
        block = block_lists[idx]
        is_redundant = True
        for ts in itertools.combinations(block, t):
            ts_id = t_subset_to_id[ts]
            if cover_count[ts_id] == 1:
                is_redundant = False
                break
                
        if is_redundant:
            # Remove this block: decrement counts
            for ts in itertools.combinations(block, t):
                ts_id = t_subset_to_id[ts]
                cover_count[ts_id] -= 1
        else:
            new_selected_indices.append(idx)
            
    selected_indices = new_selected_indices
    
    # Return the blocks as lists of integers
    return [block_lists[i] for i in selected_indices]
