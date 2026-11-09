import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute point masks as uint64 arrays
    n_words = (n_t + 63) // 64
    point_masks_u64 = np.zeros((v, n_words), dtype=np.uint64)
    
    for x in range(v):
        for sub in all_t_subsets:
            if x in sub:
                idx = t_sub_to_idx[sub]
                word_idx = idx // 64
                bit_idx = idx % 64
                point_masks_u64[x, word_idx] |= (1 << bit_idx)
                
    # Precompute block masks as uint64 arrays
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_masks_u64 = np.zeros((n_blocks, n_words), dtype=np.uint64)
    
    for i, block in enumerate(all_blocks):
        # OR the point masks
        for p in block:
            block_masks_u64[i] |= point_masks_u64[p]
            
    # Mask for valid bits in the last word
    last_word_mask = (1 << (n_t % 64)) - 1 if n_t % 64 != 0 else (2**64 - 1)
    if n_t % 64 == 0:
         last_word_mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    else:
         last_word_mask = np.uint64((1 << (n_t % 64)) - 1)

    # State
    covered_u64 = np.zeros(n_words, dtype=np.uint64)
    blocks = []
    
    while True:
        if time.time() > deadline:
            break
            
        # Check if all covered
        # covered_u64 should have all bits set in valid positions
        # Create a full mask
        full_mask = np.full(n_words, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        if n_words > 0:
            full_mask[-1] = last_word_mask
            
        if np.array_equal(covered_u64, full_mask):
            break
            
        # Find uncovered bits
        uncovered_u64 = np.bitwise_and(np.bitwise_not(covered_u64), full_mask)
        
        # Calculate gains for all blocks
        # Anded = np.bitwise_and(block_masks_u64, uncovered_u64)
        # Counts = np.count_nonzero(Anded, axis=1)
        
        # Optimization: We can skip blocks that are subsets of already covered? No.
        
        Anded = np.bitwise_and(block_masks_u64, uncovered_u64)
        Counts = np.count_nonzero(Anded, axis=1)
        
        best_idx = np.argmax(Counts)
        best_gain = Counts[best_idx]
        
        if best_gain == 0:
            break
            
        # Update covered
        covered_u64 |= block_masks_u64[best_idx]
        
        # Add block
        blocks.append(list(all_blocks[best_idx]))
        
    return blocks
