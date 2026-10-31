import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Map t-subsets to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # Precompute masks for all k-subsets
    k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(k_subs)
    
    block_masks = [0] * n_k_subs
    for i, block in enumerate(k_subs):
        mask = 0
        # Generate all t-subsets within this block
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # Initial uncovered mask: all bits set
    # Since n_t_subs can be up to 3876, 1 << 3876 is a huge int, but Python handles it.
    uncovered_mask = (1 << n_t_subs) - 1
    
    selected_blocks = []
    
    # Greedy Set Cover
    while uncovered_mask:
        if time.time() > deadline:
            break
            
        best_gain = 0
        best_idx = -1
        
        # Iterate through all blocks to find the one with max gain
        # Optimization: If best_gain reaches max possible (t-subsets in a block), we can break?
        # Max possible is C(k, t).
        max_possible_gain = 1
        for _ in range(t):
            pass # Just calculating C(k,t)
        # C(7,4) = 35
        max_gain_val = 35
        
        for i in range(n_k_subs):
            # If this block is already selected, skip? 
            # No, we can re-select? No, blocks must be distinct.
            # But if we select a block, we update uncovered_mask.
            # The gain of a previously selected block is 0 for the remaining uncovered?
            # Actually, we should not consider already selected blocks.
            # We can maintain a set of selected indices.
            
            # However, iterating all 50k blocks is fine.
            # To speed up, we can check if block_masks[i] & uncovered_mask is 0.
            # If 0, gain is 0.
            
            # Simple loop
            gain_mask = block_masks[i] & uncovered_mask
            if gain_mask:
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if best_gain == max_gain_val:
                        break
        
        if best_idx == -1:
            # Should not happen if uncovered_mask is not 0 and we have all blocks
            break
            
        selected_blocks.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # Minimalization
    # Check if any selected block is redundant
    # We do this by iterating in reverse order of addition (last added are most likely redundant)
    # Or just iterate all.
    # For each block, check if it covers any t-subset that is NOT covered by other selected blocks.
    
    # To do this efficiently:
    # Let covered_by_others_mask be the OR of masks of all other selected blocks.
    # This is O(N^2) if done naively. N~150. 150^2 = 22500.
    # Bitwise OR is fast.
    
    # Let's build the OR of all selected blocks first?
    # No, we need to check each block against the rest.
    
    # Alternative:
    # For each block b in selected_blocks:
    #   others_mask = 0
    #   for b2 in selected_blocks:
    #       if b2 != b:
    #           others_mask |= block_masks[b2]
    #   if block_masks[b] & ~others_mask == 0:
    #       # b is redundant
    #       remove b
    
    # This is O(N^2). 22500 iterations of OR and AND. Very fast.
    
    # We can optimize by maintaining a count of coverage?
    # But bitsets are easier.
    
    # Let's do the O(N^2) check.
    final_blocks = []
    for i, b_idx in enumerate(selected_blocks):
        redundant = True
        # Check if b_idx is needed
        # Calculate mask of all other blocks
        others_mask = 0
        for j, b2_idx in enumerate(selected_blocks):
            if i != j:
                others_mask |= block_masks[b2_idx]
        
        # If any bit in block_masks[b_idx] is not in others_mask, it's not redundant
        if block_masks[b_idx] & ~others_mask:
            redundant = False
            final_blocks.append(b_idx)
            
    # Convert indices back to lists
    result = [list(k_subs[i]) for i in final_blocks]
    
    return result
