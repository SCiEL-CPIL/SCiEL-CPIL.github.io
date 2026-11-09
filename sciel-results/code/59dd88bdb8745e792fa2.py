import itertools
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a greedy set cover approach 
    with precomputed block coverage bitmasks.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    block_masks = []
    all_blocks = []
    
    # Generate all k-blocks
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # 3. Greedy Set Cover
    # Current uncovered mask: all bits set to 1
    current_uncovered = (1 << num_t_subsets) - 1
    
    chosen_blocks = []
    used_block_indices = set()
    
    # Helper function for bit count
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    while current_uncovered != 0:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_block_idx = -1
        
        # Find the block that covers the most currently uncovered t-subsets
        for i in range(len(block_masks)):
            if i in used_block_indices:
                continue
                
            intersection = block_masks[i] & current_uncovered
            gain = count_bits(intersection)
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1:
            break
            
        chosen_blocks.append(list(all_blocks[best_block_idx]))
        used_block_indices.add(best_block_idx)
        
        current_uncovered &= ~block_masks[best_block_idx]
        
    return chosen_blocks
