import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_sub_map = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute bitmasks for all possible k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    
    block_data = [] # List of (block_tuple, mask_int)
    block_to_ids = {} # Dict: block_tuple -> list of t_subset indices
    
    for block in all_blocks:
        mask = 0
        ids = []
        for sub in itertools.combinations(block, t):
            idx = t_sub_map[sub]
            mask |= (1 << idx)
            ids.append(idx)
        block_data.append((block, mask))
        block_to_ids[block] = ids

    best_blocks = None
    best_count = float('inf')
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        blocks = _greedy_construct_bitset(v, k, t, all_t_subsets, block_data, random_seed=attempts)
        
        # Local Search: Remove redundant blocks
        blocks = _remove_redundant_fast(blocks, block_to_ids, num_t)
        
        # Second pass of removal if time permits
        if time.time() < deadline - 5.0:
             blocks = _remove_redundant_fast(blocks, block_to_ids, num_t)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
            
        if time.time() > deadline:
            break
            
    return best_blocks if best_blocks else []

def _greedy_construct_bitset(v, k, t, all_t_subsets, block_data, random_seed=0):
    random.seed(random_seed)
    
    # Initialize uncovered as a bitmask with all bits set
    uncovered = (1 << len(all_t_subsets)) - 1
    blocks = []
    
    # Create a shuffled copy of block_data for this attempt
    shuffled_data = block_data[:]
    random.shuffle(shuffled_data)
    
    while uncovered:
        best_gain = -1
        best_block = None
        best_mask = None
        
        # Scan all blocks to find the one with max marginal gain
        for block, mask in shuffled_data:
            gain = (uncovered & mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_mask = mask
        
        if best_gain == 0:
            break
            
        blocks.append(best_block)
        uncovered &= ~best_mask
        
    return blocks

def _remove_redundant_fast(blocks, block_to_ids, num_t):
    if not blocks:
        return blocks
    
    counts = [0] * num_t
    
    for block in blocks:
        for idx in block_to_ids[block]:
            counts[idx] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            redundant = True
            for idx in block_to_ids[block]:
                if counts[idx] < 2:
                    redundant = False
                    break
            if redundant:
                # Remove block
                for idx in block_to_ids[block]:
                    counts[idx] -= 1
                changed = True
            else:
                new_blocks.append(block)
        blocks = new_blocks
        
    return blocks
