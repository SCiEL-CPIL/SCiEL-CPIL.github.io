import itertools
import time
import random
from math import comb

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using an optimized greedy algorithm 
    with a post-processing reduction phase.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    # 1. Pre-compute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_index = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # 2. Pre-compute all k-blocks and their coverage masks
    # A mask is an integer where bit i is 1 if the i-th t-subset is covered
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in all_k_blocks:
        mask = 0
        # Generate all t-subsets of this block
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_index[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # 3. Greedy Set Cover
    # Initial state: all t-subsets are uncovered
    all_ones_mask = (1 << num_t_subsets) - 1
    uncovered_mask = all_ones_mask
    
    selected_blocks = []
    selected_indices = []
    
    # We need to efficiently find the block that covers the most uncovered subsets.
    # Naive: O(N_blocks * cost_popcount). N_blocks ~ 50k. Popcount is fast.
    # 50k * 100 iterations = 5M popcounts. This should be fast enough in Python?
    # Python loop overhead might be high. Let's try to optimize or use numpy if possible?
    # Numpy isn't strictly required but available. However, bit operations on Python ints are C-speed.
    # The loop over 50k blocks in Python might take ~0.05s per iteration. 100 iterations = 5s.
    # This is acceptable.
    
    while uncovered_mask and time.time() < deadline:
        best_gain = 0
        best_idx = -1
        
        # Find the block with maximum coverage of uncovered subsets
        for i in range(len(all_k_blocks)):
            # Count bits set in (block_mask & uncovered_mask)
            gain = (block_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                # Early termination if we found a block that covers all remaining?
                # Max possible gain is bit_count(uncovered_mask).
                # If best_gain == uncovered_mask.bit_count(), we can break.
                # But usually we don't find a perfect match immediately.
        
        if best_idx == -1:
            break # Should not happen if uncovered_mask is not 0 and blocks exist
            
        selected_indices.append(best_idx)
        selected_blocks.append(list(all_k_blocks[best_idx]))
        uncovered_mask &= ~block_masks[best_idx]
        
    if not selected_blocks:
        return []

    # 4. Post-processing: Remove redundant blocks
    # A block is redundant if all subsets it covers are covered by the union of other blocks.
    # i.e., mask_i & ~(OR_{j!=i} mask_j) == 0
    
    def reduce_blocks(indices):
        if len(indices) <= 1:
            return indices
            
        masks = [block_masks[i] for i in indices]
        n = len(indices)
        
        # Calculate OR of all masks
        total_or = 0
        for m in masks:
            total_or |= m
            
        # Pass 1: Remove blocks that are subsets of the OR of others
        # Note: If two blocks are identical, one is redundant.
        # If block A's subsets are all in block B, A is redundant.
        
        # Efficient check:
        # For each i, check if masks[i] & ~(total_or ^ masks[i]) == 0?
        # No, total_or ^ masks[i] is not "OR of others" if masks overlap.
        # "OR of others" is not easily computed from total_or.
        
        # Alternative: 
        # For each i, let others_or = 0.
        # This is O(N^2). N is small (~150). 150*150 = 22500 ops. Fast.
        
        to_remove = set()
        for i in range(n):
            # Check if masks[i] is covered by union of masks[j] for j != i
            # We can approximate or do exact.
            # Exact:
            union_others = 0
            for j in range(n):
                if i != j:
                    union_others |= masks[j]
            
            # If all bits in masks[i] are in union_others, it's redundant
            if (masks[i] & ~union_others) == 0:
                to_remove.add(i)
                
        if to_remove:
            new_indices = [indices[i] for i in range(n) if i not in to_remove]
            return reduce_blocks(new_indices)
            
        return indices

    # Apply reduction
    if time.time() < deadline:
        selected_indices = reduce_blocks(selected_indices)
        selected_blocks = [all_k_blocks[i] for i in selected_indices]

    return selected_blocks
