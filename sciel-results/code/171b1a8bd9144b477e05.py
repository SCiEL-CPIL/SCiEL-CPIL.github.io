import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    # Leave 20s buffer for cleanup and safety
    deadline = start_time + 100.0
    
    # 1. Precompute T-sets and mapping
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}
    num_tsets = len(all_tsets)
    
    # 2. Precompute Blocks and their T-set masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    
    # Pre-compute the list of t-set indices for each block?
    # No, we need the mask for bitwise ops.
    # But for the removal phase, we need to iterate t-sets of a block.
    # We can store the list of t-set indices for each block to speed up removal.
    block_tset_indices = [[] for _ in range(num_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        indices = []
        for tsub in itertools.combinations(block, t):
            idx = tset_to_idx[tsub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_tset_indices[i] = indices
        
    # 3. Greedy Construction
    uncovered_mask = (1 << num_tsets) - 1
    selected_block_indices = []
    
    while uncovered_mask:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_idx = -1
        max_possible_gain = 1 # C(k,t)
        # Calculate C(k,t)
        max_possible_gain = 1
        for i in range(1, t+1):
            max_possible_gain = (max_possible_gain * (k - t + i)) // i
            
        for i in range(num_blocks):
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == max_possible_gain:
                    break
        
        if best_idx == -1:
            # Should not happen
            break
            
        selected_block_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # 4. Removal Phase (Polish)
    # We can run this multiple times if time permits
    # We need cover_counts
    
    def run_removal_pass(indices):
        if time.time() > deadline - 5:
            return indices
            
        cover_counts = [0] * num_tsets
        
        for idx in indices:
            for t_idx in block_tset_indices[idx]:
                cover_counts[t_idx] += 1
                
        new_indices = []
        # Iterate in reverse order (later blocks are more likely redundant)
        for idx in reversed(indices):
            if time.time() > deadline - 2:
                new_indices.append(idx)
                continue
                
            is_redundant = True
            for t_idx in block_tset_indices[idx]:
                if cover_counts[t_idx] <= 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                for t_idx in block_tset_indices[idx]:
                    cover_counts[t_idx] -= 1
            else:
                new_indices.append(idx)
                
        return new_indices

    # Run removal pass 1
    selected_block_indices = run_removal_pass(selected_block_indices)
    
    # Run removal pass 2 if time allows
    if time.time() < deadline - 10:
        selected_block_indices = run_removal_pass(selected_block_indices)
        
    # Run removal pass 3 if time allows
    if time.time() < deadline - 15:
        selected_block_indices = run_removal_pass(selected_block_indices)

    # 5. Convert indices to blocks
    result_blocks = [list(all_blocks[i]) for i in selected_block_indices]
    
    return result_blocks
