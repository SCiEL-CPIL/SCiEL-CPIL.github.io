import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy Initialization followed by Local Search (Deletion + Swap).
    """
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute t-subset indices for each block
    # This is O(num_blocks * C(k,t))
    block_t_indices = np.zeros((num_blocks, int(len(list(itertools.combinations(range(k), t))))), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, ts in enumerate(itertools.combinations(block, t)):
             block_t_indices[i, j] = t_sub_to_idx[ts]
             
    # Helper to get t-indices for a block tuple
    # We store block index in a set for O(1) lookup of "is this block in the covering"
    
    # 2. Greedy Initialization
    uncovered = np.ones(num_t_subs, dtype=np.int8)
    blocks_used = []
    used_block_mask = np.zeros(num_blocks, dtype=bool)
    
    while uncovered.sum() > 0:
        # Calculate gains
        gains = uncovered[block_t_indices].sum(axis=1)
        gains[used_block_mask] = -1
        best_idx = np.argmax(gains)
        
        if gains[best_idx] <= 0:
            break
            
        blocks_used.append(best_idx) # Store indices for efficiency
        used_block_mask[best_idx] = True
        uncovered[block_t_indices[best_idx]] = 0
        
    # 3. Local Search
    # State: list of block indices, set of block indices, coverage counts
    current_blocks = list(blocks_used)
    current_block_set = set(current_blocks)
    
    # Initialize coverage counts
    t_sub_count = np.zeros(num_t_subs, dtype=np.int32)
    for b_idx in current_blocks:
        t_sub_count[block_t_indices[b_idx]] += 1
        
    improved = True
    max_iterations = 1000 # Safety cap
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        
        # Phase 1: Deletion
        # Try to delete blocks
        # To be efficient, we can iterate and check if min count of block's t-subs > 1
        for i in range(len(current_blocks) - 1, -1, -1):
            b_idx = current_blocks[i]
            idxs = block_t_indices[b_idx]
            if np.min(t_sub_count[idxs]) > 1:
                # Remove
                t_sub_count[idxs] -= 1
                current_blocks.pop(i)
                current_block_set.remove(b_idx)
                improved = True
                break # Restart deletion loop as counts changed
                
        if improved:
            continue
            
        # Phase 2: Swap / Perturbation
        # Try to swap a block to find a better configuration that allows future deletions
        # We perform a limited number of random swaps
        swaps_attempted = 0
        max_swaps = 500
        
        while swaps_attempted < max_swaps:
            swaps_attempted += 1
            if not current_blocks:
                break
                
            # Pick a random block to swap out
            i = random.randint(0, len(current_blocks) - 1)
            old_idx = current_blocks[i]
            
            # Pick a random block to swap in
            # We need a block not in current_block_set
            # Efficient way: pick random index, check if in set. If so, retry.
            # Or maintain a list of available blocks? That's expensive to update.
            # Given num_blocks ~ 75k, random sampling is fine.
            new_idx = -1
            for _ in range(10): # Try a few times to find an unused block
                cand = random.randint(0, num_blocks - 1)
                if cand not in current_block_set:
                    new_idx = cand
                    break
            
            if new_idx == -1:
                continue
                
            # Check if swap maintains validity
            # Temporarily remove old, add new
            t_sub_count[block_t_indices[old_idx]] -= 1
            t_sub_count[block_t_indices[new_idx]] += 1
            
            # Check if any t-subset count is 0
            if np.any(t_sub_count == 0):
                # Revert
                t_sub_count[block_t_indices[new_idx]] -= 1
                t_sub_count[block_t_indices[old_idx]] += 1
            else:
                # Accept swap
                current_blocks[i] = new_idx
                current_block_set.remove(old_idx)
                current_block_set.add(new_idx)
                improved = True # Signal that state changed, might allow deletion next loop
                break # Break out of swap loop, go back to deletion phase
                
    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in current_blocks]
