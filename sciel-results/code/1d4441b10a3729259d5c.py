import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_set = set(all_t_subs)
    
    # Precompute bitmasks for each t-subset and each block
    # This allows O(1) intersection checks via bitwise AND
    # t_sub_to_idx: maps t-subset tuple to its index
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # Precompute all k-blocks and their coverage bitmasks
    # A block covers a t-subset if the t-subset is a subset of the block.
    # We represent the set of covered t-subsets as a Python integer (bitmask).
    # Since C(19,5) = 11628, this requires ~11628 bits, which fits in a Python int.
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    for b in all_k_blocks:
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = t_sub_to_idx[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Also precompute the "uncovered" mask for quick updates
    # Full mask with all bits set
    full_mask = (1 << len(all_t_subs)) - 1
    
    best_blocks = None
    best_size = float('inf')
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        blocks = greedy_cover_fast(v, k, t, all_k_blocks, block_masks, full_mask, random_seed=attempts)
        
        if blocks:
            blocks = reduce_blocks_fast(blocks, t, all_t_subs, t_sub_to_idx, block_masks)
            
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # If we get very close to the lower bound, stop early?
        # Lower bound is 264. Best known 335.
        # If we reach 300, it's a significant improvement, but we can keep trying.
        if best_size <= 300:
            break

    if best_blocks is None:
        best_blocks = greedy_cover_fast(v, k, t, all_k_blocks, block_masks, full_mask, random_seed=0)
        best_blocks = reduce_blocks_fast(best_blocks, t, all_t_subs, t_sub_to_idx, block_masks)
        
    return [list(b) for b in best_blocks]

def greedy_cover_fast(v, k, t, all_k_blocks, block_masks, full_mask, random_seed=0):
    rng = random.Random(random_seed)
    uncovered_mask = full_mask
    blocks = []
    
    # To speed up the search for the best block, we can maintain a list of candidate blocks
    # or just iterate through all blocks.
    # With ~75k blocks, iterating is fast if we use bitwise ops.
    
    while uncovered_mask:
        best_block_idx = -1
        best_gain = 0
        
        # Iterate through all blocks to find the one covering the most uncovered t-subsets
        # We can optimize by skipping blocks that are subsets of already chosen blocks?
        # Or just brute force the gain calculation.
        # gain = popcount(uncovered_mask & block_mask[i])
        
        for i, mask in enumerate(block_masks):
            gain = bin(uncovered_mask & mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                if gain == 56: # Max possible gain for C(8,5)
                    break
        
        if best_block_idx == -1:
            break
            
        blocks.append(all_k_blocks[best_block_idx])
        uncovered_mask &= ~block_masks[best_block_idx]
        
    return blocks

def reduce_blocks_fast(blocks, t, all_t_subs, t_sub_to_idx, block_masks):
    if not blocks:
        return []
        
    # Convert blocks to indices in all_k_blocks? 
    # We don't have a direct map from block tuple to index in all_k_blocks in the fast version easily without building one.
    # Let's build a map: block_tuple -> index
    # But we passed all_k_blocks. Let's just use the masks directly.
    # We need to know which blocks cover which t-subsets.
    
    # Re-compute coverage counts using bitmasks?
    # It's easier to use the standard reduction if we have the t-subsets.
    # But we can optimize the check.
    
    # Let's stick to a slightly optimized standard reduction using sets of t-subsets for each block
    # Or better, just use the standard reduction but with precomputed t-subsets per block if possible.
    # Given the time, the standard reduction is likely fast enough for < 400 blocks.
    
    # Map block tuple to its t-subsets
    block_tsubs = {}
    for b in blocks:
        b_tuple = tuple(b)
        block_tsubs[b_tuple] = list(itertools.combinations(b, t))
        
    counts = {}
    for b in blocks:
        b_tuple = tuple(b)
        for sub in block_tsubs[b_tuple]:
            counts[sub] = counts.get(sub, 0) + 1
            
    changed = True
    current_blocks = [list(b) for b in blocks]
    
    while changed:
        changed = False
        new_blocks = []
        for b in current_blocks:
            b_tuple = tuple(b)
            is_redundant = True
            for sub in block_tsubs[b_tuple]:
                if counts.get(sub, 0) == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_blocks.append(b)
            else:
                changed = True
                for sub in block_tsubs[b_tuple]:
                    counts[sub] -= 1
        current_blocks = new_blocks
        
    return current_blocks
