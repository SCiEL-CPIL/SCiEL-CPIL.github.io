import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    rng = random.Random(42)
    
    # 1. Precompute all t-subsets
    # Map each t-subset to a unique integer index
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute coverage masks for all possible k-blocks
    # A block is a k-subset. It covers all t-subsets contained within it.
    # We represent the set of covered t-subsets as a bitmask (Python int).
    # Bit i is 1 if t-subset i is covered.
    
    # Note: n_t_subs is 3876. Python ints can handle this easily.
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_masks[i] is the bitmask of t-subsets covered by block i
    block_masks = [0] * n_k_blocks
    
    # Map from block tuple to its index in all_k_blocks for quick lookup if needed
    # But we just need the list of blocks and their masks.
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        # Iterate over all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 3. Greedy Construction
    uncovered_mask = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    # To speed up greedy, we can sort blocks by initial coverage (all same) 
    # or just iterate.
    # We will iterate through all blocks to find the best one each time.
    
    # Precompute block_masks as a list of ints for fast bit operations
    
    while uncovered_mask != 0:
        if time.time() > deadline - 30:
            break
            
        best_idx = -1
        best_gain = -1
        
        # Optimization: Skip blocks that are already selected?
        # No, we need to find the best *new* block.
        # We can maintain a list of active block indices or just iterate all.
        # Iterating all 50k is fast enough.
        
        for i in range(n_k_blocks):
            # Calculate gain: how many currently uncovered subsets does this block cover?
            # gain = popcount(block_masks[i] & uncovered_mask)
            gain = (block_masks[i] & uncovered_mask).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                # Early exit if we found a perfect match (covers all remaining)? Unlikely.
                # If gain == 0, we can break if we've checked some, but best_gain starts -1.
        
        if best_idx == -1 or best_gain == 0:
            # Should not happen if uncovered_mask != 0
            break
            
        selected_block_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # 4. Convert indices back to blocks
    blocks = [list(all_k_blocks[i]) for i in selected_block_indices]
    
    # 5. Local Search: Redundancy Removal
    # We want to remove any block that is not essential.
    # A block is essential if it is the *only* block covering at least one t-subset.
    
    # Precompute cover counts for each t-subset
    cover_count = [0] * n_t_subs
    
    # For each selected block, increment count for its t-subsets
    selected_masks = [block_masks[i] for i in selected_block_indices]
    
    for mask in selected_masks:
        # Iterate over set bits in mask
        m = mask
        while m:
            # Get index of lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
            
    # Pass 1: Remove non-critical blocks
    # We iterate through blocks and check if any t-subset in it has cover_count == 1.
    # If not, the block can be removed.
    
    # To make this efficient, we can maintain the cover_count and update it.
    # However, removing a block changes cover_counts for all other blocks' subsets.
    # A simple approach:
    # 1. Build a list of (block_idx, mask).
    # 2. Iterate. For each block, check if it's critical.
    # 3. If not, remove it and decrement cover_counts for its subsets.
    # 4. Repeat until no more removals.
    
    # Since the number of blocks is small (~150), we can afford to be slightly inefficient.
    
    current_blocks = selected_block_indices[:]
    current_masks = [block_masks[i] for i in current_blocks]
    
    # Re-initialize cover_count for the current set
    cover_count = [0] * n_t_subs
    for mask in current_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
            
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        new_current_blocks = []
        new_current_masks = []
        
        for i in range(len(current_blocks)):
            mask = current_masks[i]
            # Check if this block is the only cover for any subset
            is_critical = False
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    is_critical = True
                    break
                m ^= lsb
                
            if is_critical:
                new_current_blocks.append(current_blocks[i])
                new_current_masks.append(mask)
            else:
                # Remove this block
                # Decrement cover counts
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                improved = True
                
        current_blocks = new_current_blocks
        current_masks = new_current_masks
    
    # 6. Swap-based Local Search
    # Try to replace one block with another that allows removing more blocks
    # or try to find a better combination
    
    # Precompute for each block, the list of t-subsets it covers (for faster access)
    block_t_subs = {}
    for idx in current_blocks:
        block = all_k_blocks[idx]
        subs = list(itertools.combinations(block, t))
        block_t_subs[idx] = [sub_to_idx[s] for s in subs]
    
    # Try swaps: remove block i, add block j, then do redundancy removal
    # Keep track of best solution found
    best_count = len(current_blocks)
    best_blocks = current_blocks[:]
    
    # Precompute masks for all blocks as numpy arrays for faster operations?
    # No, Python ints with bit_count are fast enough.
    
    # Strategy: 
    # 1. For each block in current solution, try removing it and adding a random other block
    # 2. Do redundancy removal
    # 3. If we get a smaller solution, keep it
    
    # To be efficient, we'll do multiple rounds of swap attempts
    
    max_swap_rounds = 50
    for round_num in range(max_swap_rounds):
        if time.time() > deadline - 5:
            break
            
        if len(current_blocks) <= best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]
        
        # Try a few random swaps
        n_swaps_per_round = 10
        for _ in range(n_swaps_per_round):
            if time.time() > deadline - 5:
                break
                
            if len(current_blocks) < 2:
                break
                
            # Pick a random block to remove
            remove_idx_pos = rng.randrange(len(current_blocks))
            remove_block_idx = current_blocks[remove_idx_pos]
            
            # Pick a random block to add (not already in solution)
            # For efficiency, just pick a random block from all_k_blocks
            # and check if it's already there
            new_block_idx = rng.randrange(n_k_blocks)
            
            # Check if new_block_idx is already in current_blocks
            # This is O(n) but n is small (~150)
            if new_block_idx in current_blocks:
                continue
                
            # Create a new solution: remove remove_block_idx, add new_block_idx
            new_blocks = [b for i, b in enumerate(current_blocks) if i != remove_idx_pos]
            new_blocks.append(new_block_idx)
            
            # Do redundancy removal on new_blocks
            new_cover_count = [0] * n_t_subs
            for b in new_blocks:
                mask = block_masks[b]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    new_cover_count[idx] += 1
                    m ^= lsb
            
            # Check if all t-subsets are covered
            # If any has count 0, this solution is invalid
            valid = True
            for idx in range(n_t_subs):
                if new_cover_count[idx] == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Do redundancy removal
            improved_local = True
            while improved_local:
                improved_local = False
                new_new_blocks = []
                for b in new_blocks:
                    mask = block_masks[b]
                    # Check if this block is the only cover for any subset
                    is_critical = False
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if new_cover_count[idx] == 1:
                            is_critical = True
                            break
                        m ^= lsb
                        
                    if is_critical:
                        new_new_blocks.append(b)
                    else:
                        # Remove this block
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            new_cover_count[idx] -= 1
                            m ^= lsb
                        improved_local = True
                        
                new_blocks = new_new_blocks
            
            # If we got a smaller solution, keep it
            if len(new_blocks) < len(current_blocks):
                current_blocks = new_blocks
                current_masks = [block_masks[i] for i in current_blocks]
                # Recompute block_t_subs for new blocks
                block_t_subs = {}
                for idx in current_blocks:
                    block = all_k_blocks[idx]
                    subs = list(itertools.combinations(block, t))
                    block_t_subs[idx] = [sub_to_idx[s] for s in subs]
                
                # Update best
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = current_blocks[:]
    
    # Final blocks
    blocks = [list(all_k_blocks[i]) for i in best_blocks]
    
    return blocks
