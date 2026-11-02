import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    points = list(range(v))
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute masks for all possible blocks
    # A block is a k-subset. It covers all t-subsets within it.
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if block i covers t_subset j
    block_masks = [0] * n_blocks
    
    # Precomputation cost: O(n_blocks * C(k,t))
    # 75582 * C(8,5) = 75582 * 56 = ~4.2M operations. This is fast.
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Greedy Set Cover
    # uncovered_mask: bits set for t-subsets not yet covered
    uncovered_mask = (1 << n_t) - 1
    selected_blocks = []
    # counts[i] is the number of selected blocks covering t_subset i
    # We don't strictly need counts for greedy, but for removal phase we do.
    # We will maintain a list of selected block indices.
    
    # To speed up greedy, we can sort blocks by their total coverage size (popcount) descending?
    # No, the gain depends on current uncovered set.
    
    # Optimization: Instead of checking all blocks, we can just check all.
    # 75k blocks * ~300 iterations = 22.5M checks.
    # Each check: `gain = (uncovered_mask & block_masks[i]).bit_count()`
    # This is very fast in Python.
    
    while uncovered_mask:
        if time.time() > deadline:
            break
            
        best_block_idx = -1
        best_gain = 0
        
        # Iterate through all blocks
        for i in range(n_blocks):
            # If a block covers no uncovered elements, skip
            # gain = (uncovered_mask & block_masks[i]).bit_count()
            # We can skip if block_masks[i] is a subset of already covered? 
            # No, just compute gain.
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1:
            # Should not happen if uncovered_mask is not 0
            break
            
        # Select the block
        selected_blocks.append(best_block_idx)
        uncovered_mask &= ~block_masks[best_block_idx]
        
        # Early termination if we have too many blocks? No, we want min.
        
    # Local Search: Remove redundant blocks
    # A block is redundant if every t-subset it covers is covered by at least one other selected block.
    # We maintain a count of how many times each t-subset is covered by the current set of selected blocks.
    
    # Initialize counts
    counts = [0] * n_t
    for b_idx in selected_blocks:
        mask = block_masks[b_idx]
        # Iterate through set bits in mask
        # Efficient way to iterate bits:
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
    # Iterative removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate through selected blocks
        # We can iterate in reverse or any order
        # To make it efficient, we can check redundancy for each block
        new_selected = []
        for b_idx in selected_blocks:
            mask = block_masks[b_idx]
            redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
                
            if redundant:
                # Remove this block
                # Decrement counts for all t-subsets in this block
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_selected.append(b_idx)
                
        selected_blocks = new_selected
        
    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in selected_blocks]
