import itertools
import time
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Generate all k-blocks and their masks
    # A block covers a t-subset if the t-subset is a subset of the block
    # We can generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(k_blocks)
    
    # Precompute masks for each block
    # mask is an integer where bit j is 1 if t_subsets[j] is in the block
    block_masks = [0] * n_blocks
    # Also store the block itself for output
    # For efficiency, we might want to map block tuple to index
    
    # Optimization: Instead of checking all t_subsets for each block,
    # we can iterate over the t_subsets within the block.
    # For a block B of size k, the t-subsets inside it are combinations of B of size t.
    for i, block in enumerate(k_blocks):
        mask = 0
        # Generate all t-combinations within this block
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask

    # 3. Greedy Phase
    uncovered_mask = (1 << n_t) - 1
    selected_indices = []
    
    # We will repeatedly pick the block with max gain
    # To speed up finding max, we might just iterate all.
    # 75582 blocks.
    
    while uncovered_mask:
        if time.time() > deadline:
            break
            
        best_gain = 0
        best_idx = -1
        
        # Iterate through all blocks to find max gain
        # We can skip blocks that are already selected? No, we can select same block? 
        # "blocks must be distinct". So we should not select the same block twice.
        # We can mark selected blocks as having 0 gain or skip them.
        
        for i in range(n_blocks):
            if i in selected_indices: 
                # This check is O(N) if selected_indices is a list. 
                # Better use a set for selected.
                pass
            # Actually, just computing gain is fine. If it's selected, gain might be 0 
            # because its subsets are already covered? Not necessarily, but likely.
            # But we must not pick it again.
            
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_gain == 0:
            # This should not happen if a covering exists
            break
            
        selected_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]

    # 4. Polish Phase (Removal)
    # We have selected_indices.
    # We want to remove any block that is redundant.
    # A block is redundant if all its t-subsets are covered by other selected blocks.
    
    # Use cover counts
    cover_count = [0] * n_t
    selected_set = set(selected_indices)
    
    # Initialize counts
    for i in selected_indices:
        # Extract bits from mask
        mask = block_masks[i]
        # Iterate set bits
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            mask ^= lsb
            
    # Try to remove blocks
    # We can do multiple passes
    for _ in range(5): # Max 5 passes
        if time.time() > deadline:
            break
        removed = False
        # Iterate through a copy of selected indices
        # We should iterate in an order that might help, e.g., random or by size of overlap
        # For now, just iterate
        for i in list(selected_set):
            # Check if block i is redundant
            # Check all t-subsets in block i
            redundant = True
            mask = block_masks[i]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                if cover_count[idx] <= 1:
                    redundant = False
                    break
                mask ^= lsb
                
            if redundant:
                # Remove it
                selected_set.remove(i)
                # Decrement counts
                mask = block_masks[i]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    mask ^= lsb
                removed = True
                
        if not removed:
            break
            
    # 5. Construct Output
    blocks = [list(k_blocks[i]) for i in selected_set]
    return blocks
