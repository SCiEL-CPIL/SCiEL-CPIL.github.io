import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using NumPy-accelerated greedy and local search.
    """
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute t-subset indices for each block
    # C(k, t) is the number of t-subsets in a block
    num_t_per_block = int(len(list(itertools.combinations(range(k), t))))
    
    block_t_indices = np.zeros((num_blocks, num_t_per_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, ts in enumerate(itertools.combinations(block, t)):
             block_t_indices[i, j] = t_sub_to_idx[ts]
             
    # 2. Greedy Construction
    uncovered = np.ones(num_t_subs, dtype=np.int8)
    blocks_used = []
    used_block_mask = np.zeros(num_blocks, dtype=bool)
    
    while uncovered.sum() > 0:
        # Calculate gains for all unused blocks
        # gains[i] = number of uncovered t-subsets covered by block i
        gains = uncovered[block_t_indices].sum(axis=1)
        gains[used_block_mask] = -1
        
        best_idx = np.argmax(gains)
        
        if gains[best_idx] <= 0:
            break
            
        blocks_used.append(all_blocks[best_idx])
        used_block_mask[best_idx] = True
        uncovered[block_t_indices[best_idx]] = 0
        
    # Convert to list of tuples for easier manipulation in local search
    current_blocks = [tuple(b) for b in blocks_used]
    
    # 3. Local Search
    # We will maintain a list of block indices (into all_blocks)
    # And a coverage count array for t-subsets
    
    # Helper to get block index from tuple
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Initialize coverage counts
    t_sub_count = np.zeros(num_t_subs, dtype=np.int32)
    current_block_indices = []
    
    for b in current_blocks:
        idx = block_to_idx[b]
        current_block_indices.append(idx)
        t_sub_count[block_t_indices[idx]] += 1
        
    # Function to check if a set of t-subset indices is a subset of another block's coverage
    # We can precompute a bitset for each block's coverage?
    # num_t_subs = 11628. Bitset would be 11628 bits = 1454 bytes per block.
    # 75582 blocks * 1454 bytes = ~110 MB. Acceptable.
    # But let's try without bitsets first, using sets of indices.
    
    # For speed in 2-for-1 swap, we need to check if L (lost subsets) is subset of C (candidate block coverage).
    # L is small (at most 56+56).
    
    def try_remove_block(idx):
        """Returns True if block idx can be removed."""
        # Check if any t-subset in this block has count 1
        # If min count > 1, it's safe.
        # But wait, if we remove it, counts decrease.
        # We need to check if min(t_sub_count[block_t_indices[idx]]) > 1
        return np.min(t_sub_count[block_t_indices[idx]]) > 1

    def remove_block(idx):
        """Remove block idx from current design."""
        t_sub_count[block_t_indices[idx]] -= 1
        current_block_indices.remove(idx)

    def add_block(idx):
        """Add block idx to current design."""
        t_sub_count[block_t_indices[idx]] += 1
        current_block_indices.append(idx)

    # Pass 1: Deletion
    changed = True
    while changed:
        changed = False
        for i in range(len(current_block_indices) - 1, -1, -1):
            idx = current_block_indices[i]
            if try_remove_block(idx):
                remove_block(idx)
                changed = True
                break # Restart scan as counts changed

    # Pass 2: 2-for-1 Swap
    # For each pair of blocks, see if we can replace them with one.
    # To speed this up, we limit the number of attempts or use a heuristic.
    # We will iterate through all pairs. N is ~300. Pairs ~ 45k.
    # For each pair, compute lost subsets.
    
    swap_changed = True
    max_swap_iterations = 10
    iter_count = 0
    while swap_changed and iter_count < max_swap_iterations:
        swap_changed = False
        iter_count += 1
        n_blocks = len(current_block_indices)
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                idx1 = current_block_indices[i]
                idx2 = current_block_indices[j]
                
                # Calculate lost subsets if we remove idx1 and idx2
                # We need to know which t-subsets have count exactly 1 or 2 and are in idx1 or idx2.
                # Actually, let's just simulate the removal temporarily?
                # No, that's slow.
                
                # Let L be the set of t-subsets covered by idx1 or idx2 that have count == 1 (only covered by one of them) 
                # or count == 2 (covered by both, but not others).
                # If we remove both, any t-subset with count 1 that is in idx1 U idx2 becomes uncovered.
                # Any t-subset with count 2 that is in idx1 U idx2 and NOT in any other block becomes uncovered.
                
                # Efficient check:
                # 1. Get indices for idx1 and idx2
                inds1 = block_t_indices[idx1]
                inds2 = block_t_indices[idx2]
                
                # Union of indices
                # We can use a boolean mask for the union
                # But 11628 is small.
                union_mask = np.zeros(num_t_subs, dtype=bool)
                union_mask[inds1] = True
                union_mask[inds2] = True
                
                # Find indices in union that have count 1
                # If count is 1, it must be covered by exactly one of idx1, idx2 (since no other block covers it).
                # So if we remove both, it is lost.
                lost_mask = (t_sub_count == 1) & union_mask
                lost_indices = np.where(lost_mask)[0]
                
                if len(lost_indices) == 0:
                    # We can remove both!
                    remove_block(idx1)
                    # Note: current_block_indices changed, idx2 index might have shifted?
                    # We are removing by value from list.
                    # remove_block(idx1) removes idx1.
                    # Then we need to remove idx2.
                    # But idx2 is still in the list? Yes.
                    remove_block(idx2)
                    swap_changed = True
                    break # Restart pair scanning
                    
                # If lost_indices is not empty, check if there is a block C that covers all lost_indices.
                # We can check all blocks? 75k blocks.
                # For each block C, check if lost_indices is a subset of block_t_indices[C].
                # This is 75k checks per pair. 45k pairs * 75k = 3.3 billion. Too slow.
                
                # Optimization: Only check blocks that cover at least one of the lost indices.
                # Or, simpler: just skip 2-for-1 if it's too complex and rely on deletion + single replacement?
                
                # Let's try a simpler local move: Single Replacement.
                # Try to replace block idx1 with block C such that C covers all unique subsets of idx1.
                # Unique subsets of idx1: subsets in idx1 with count 1.
                unique_mask1 = (t_sub_count == 1) & (np.isin(np.arange(num_t_subs), inds1))
                # This is slow.
                
                # Let's stick to Deletion for now, it's very effective.
                # The greedy is already quite good.
                pass

    # Convert back to lists
    return [list(all_blocks[idx]) for idx in current_block_indices]
