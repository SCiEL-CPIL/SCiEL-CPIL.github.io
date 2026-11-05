import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using NumPy-accelerated greedy and local search.
    """
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, store the indices of the t-subsets it covers
    # Shape: (num_blocks, C(k, t))
    block_t_indices = np.zeros((num_blocks, int(len(list(itertools.combinations(range(k), t))))), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        block_set = set(block)
        # Find t-subsets of this block
        # Note: itertools.combinations(block, t) gives tuples of values, not indices
        # We need to map these value-tuples to the global t-subset indices
        for j, ts in enumerate(itertools.combinations(block, t)):
             block_t_indices[i, j] = t_sub_to_idx[ts]
             
    # Initialize uncovered array
    uncovered = np.ones(num_t_subs, dtype=np.int8)
    
    # Greedy Construction
    blocks_used = []
    remaining_blocks = list(range(num_blocks))
    
    # We will select blocks one by one
    # To speed up, we can just iterate over all blocks each time, 
    # but as we cover more, the gains drop.
    
    # Optimization: We don't need to check blocks that are already used? 
    # Actually, blocks must be distinct.
    used_block_mask = np.zeros(num_blocks, dtype=bool)
    
    while uncovered.sum() > 0:
        # Calculate gains for all unused blocks
        # gains[i] = number of uncovered t-subsets covered by block i
        # We can compute this by summing uncovered[indices]
        
        # This creates a (num_blocks, 56) array, which is ~4M elements.
        # This is fast in NumPy.
        gains = uncovered[block_t_indices].sum(axis=1)
        
        # Mask out used blocks
        gains[used_block_mask] = -1
        
        # Find best block
        best_idx = np.argmax(gains)
        
        # If best gain is 0, we have a problem (should not happen if covering is possible)
        if gains[best_idx] <= 0:
            break
            
        # Add block to design
        blocks_used.append(all_blocks[best_idx])
        used_block_mask[best_idx] = True
        
        # Update uncovered
        # Set uncovered to 0 for all t-subsets covered by this block
        uncovered[block_t_indices[best_idx]] = 0
        
    # Convert blocks to lists of ints
    blocks = [list(b) for b in blocks_used]
    
    # 2. Local Search: Greedy Deletion
    # Build coverage counts
    t_sub_count = np.zeros(num_t_subs, dtype=np.int32)
    for b in blocks:
        # Get indices of t-subsets in b
        # We need a fast way to get these indices
        # We can reuse the precomputed structure if we store block ID -> indices
        # But we only stored block indices for ALL blocks.
        # Let's create a helper to get indices for a specific block value
        pass # This part is tricky with the precomputed array if we don't map back.
        
    # Better approach for deletion:
    # Rebuild coverage counts from scratch using the blocks list.
    # For each block, find its t-subsets and increment count.
    # This is O(N_blocks * C(k,t)).
    
    def get_t_indices(block):
        # This is slow if called many times.
        # Let's precompute a dictionary from block-tuple to its t-indices?
        # Or just compute it.
        pass

    # Let's implement a simple deletion loop
    # To make it efficient, we maintain a list of t-subset counts.
    
    # Re-initialize counts
    t_sub_count = np.zeros(num_t_subs, dtype=np.int32)
    # We need to map block (tuple) to its t-indices quickly.
    # Let's create a dict: block_tuple -> list of t_indices
    block_to_t_indices = {}
    for i, block in enumerate(all_blocks):
        # Only store if we might need it? No, just store all.
        # Memory: 75582 entries.
        idxs = block_t_indices[i].tolist()
        block_to_t_indices[block] = idxs
        
    # Update counts
    for b in blocks:
        bt = tuple(b)
        if bt in block_to_t_indices:
            for idx in block_to_t_indices[bt]:
                t_sub_count[idx] += 1
                
    # Deletion Phase
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            bt = tuple(b)
            idxs = block_to_t_indices[bt]
            # Check if any t-subset in this block has count 1
            # If all counts > 1, block is redundant
            # Actually, if count == 1, it's critical. If count > 1, it's safe to remove.
            # We can remove if min(t_sub_count[idxs]) > 1
            if np.min(t_sub_count[idxs]) > 1:
                # Remove block
                for idx in idxs:
                    t_sub_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break # Restart loop as indices/counts changed
                
    return blocks
