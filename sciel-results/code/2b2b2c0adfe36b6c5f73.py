<<<<<<< SEARCH
def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, seed, deadline):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    n_blocks = len(all_blocks)
    
    # Pre-compute list of block indices for shuffling
    indices = list(range(n_blocks))
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Shuffle the order of block evaluation to diversify the greedy path
        rng.shuffle(indices)
        
        best_block_idx = -1
        max_gain = 0
        
        for idx in indices:
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block_idx = idx
                # Early exit if we found a block that covers all remaining? 
                # Unlikely for large uncovered, but if max_gain == uncovered.bit_count(), we can stop searching for better
                if max_gain == uncovered.bit_count():
                    break
            
        if best_block_idx == -1:
            # Should not happen if covering is possible
            break
            
        blocks_used.append(best_block_idx)
        uncovered &= ~block_masks[best_block_idx]
        
    return blocks_used
=======
def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, seed, deadline):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    n_blocks = len(all_blocks)
    
    # Pre-compute list of block indices
    indices = list(range(n_blocks))
    # Shuffle once to diversify the path for this restart
    rng.shuffle(indices)
    
    while uncovered:
        if time.time() > deadline:
            return None # Signal timeout/invalid
            
        best_block_idx = -1
        max_gain = 0
        
        for idx in indices:
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block_idx = idx
                # Early exit if we found a block that covers all remaining? 
                if max_gain == uncovered.bit_count():
                    break
            
        if best_block_idx == -1:
            # Should not happen if covering is possible
            return None
            
        blocks_used.append(best_block_idx)
        uncovered &= ~block_masks[best_block_idx]
        
    return blocks_used
>>>>>>> REPLACE

