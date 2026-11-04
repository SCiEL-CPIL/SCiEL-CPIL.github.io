import itertools
import time
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a greedy heuristic followed by 
    iterative redundancy pruning based on coverage counts.
    """
    deadline = time.monotonic() + 115.0  # Safety margin under 120s
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Generate all k-blocks
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Precompute for each k-block:
    # - The bitmask of t-subsets it covers
    # - The list of t-subset indices it covers (for count updates)
    block_masks = [0] * num_kblocks
    block_tsub_indices = [None] * num_kblocks
    
    for i, b in enumerate(all_kblocks):
        mask = 0
        indices = []
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_tsub_indices[i] = indices
        
    full_mask = (1 << num_tsubs) - 1
    
    # 2. Greedy Construction
    selected_blocks = []
    uncovered_mask = full_mask
    
    # Optimization: Keep a list of candidate block indices. 
    # In a full greedy, we check all blocks every time. 
    # With ~50k blocks and ~150 iterations, this is 7.5 million checks.
    # Each check is a bitwise AND and popcount. This is fast in Python if we use integers.
    
    while uncovered_mask:
        if time.monotonic() > deadline:
            break
            
        best_gain = 0
        best_block_idx = -1
        
        # Find block with max new coverage
        for i in range(num_kblocks):
            # Only consider if it covers at least one new bit
            # quick check: if mask & uncovered_mask == 0, gain is 0
            common = block_masks[i] & uncovered_mask
            if common:
                # Python 3.10+ has int.bit_count(), older use bin(x).count('1')
                # Assuming modern Python environment for speed
                gain = common.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    
        if best_gain == 0:
            # Should not happen if uncovered_mask is not 0, unless no blocks left? 
            # But we have all blocks available. This implies a bug or impossible task.
            break
            
        # Select the best block
        selected_blocks.append(best_block_idx)
        
        # Update uncovered mask
        uncovered_mask &= ~block_masks[best_block_idx]
        
        # Early exit if all covered
        if uncovered_mask == 0:
            break
            
    # 3. Redundancy Pruning
    # We want to remove blocks that are not essential.
    # A block is redundant if every t-subset it covers is covered by at least one other block.
    
    # Initialize counts
    counts = [0] * num_tsubs
    for b_idx in selected_blocks:
        for t_idx in block_tsub_indices[b_idx]:
            counts[t_idx] += 1
            
    # Iterative removal
    # We iterate through the list of selected blocks. 
    # If a block is redundant, we remove it and update counts.
    # We must be careful with indices if we remove from the list while iterating.
    # Easier: Use a boolean array 'active' or build a new list.
    # Given the small number of blocks (~150), building a new list is fine.
    
    changed = True
    while changed:
        if time.monotonic() > deadline:
            break
        changed = False
        new_selected = []
        for b_idx in selected_blocks:
            # Check if this block is redundant
            redundant = True
            for t_idx in block_tsub_indices[b_idx]:
                if counts[t_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it: decrement counts
                for t_idx in block_tsub_indices[b_idx]:
                    counts[t_idx] -= 1
                changed = True
            else:
                new_selected.append(b_idx)
        
        selected_blocks = new_selected
        
    # 4. Final Output
    # Convert indices back to lists of points
    final_blocks = [list(all_kblocks[i]) for i in selected_blocks]
    
    return final_blocks
