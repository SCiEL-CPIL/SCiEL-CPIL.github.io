import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Map 5-subsets to IDs
    points = range(12)
    all_5_subsets = [frozenset(s) for s in combinations(points, 5)]
    subset_to_id = {s: i for i, s in enumerate(all_5_subsets)}
    
    # 2. Generate all 6-subsets (blocks)
    all_6_subsets = [tuple(s) for s in combinations(points, 6)]
    
    # 3. Precompute masks for each 6-subset
    block_5_ids = [] # list of lists of 5-subset IDs
    block_masks = [] # list of ints
    
    for b in all_6_subsets:
        ids = []
        mask = 0
        for i in range(6):
            s = frozenset(b[:i] + b[i+1:])
            id_ = subset_to_id[s]
            ids.append(id_)
            mask |= (1 << id_)
        block_5_ids.append(ids)
        block_masks.append(mask)
        
    # candidates_for_5: for each 5-subset ID, which blocks contain it?
    candidates_for_5 = [[] for _ in range(792)]
    for b_idx, ids in enumerate(block_5_ids):
        for id_ in ids:
            candidates_for_5[id_].append(b_idx)
            
    ALL_MASK = (1 << 792) - 1
    
    # Fix first block: (0, 1, 2, 3, 4, 5)
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6_subsets.index(first_block)
    
    # Fix second block: (0, 1, 2, 3, 4, 6)
    # This breaks significant symmetry.
    # We must ensure this block is valid with the first (it is, they share 5 points? No, they share 0,1,2,3,4.
    # Wait, S(5,6,12) blocks intersect in at most 5 points?
    # Any two blocks intersect in 0, 1, 2, 3, 4, or 5 points?
    # In S(5,6,12), any 5 points are in exactly one block.
    # If two blocks share 5 points, they are the same block.
    # So distinct blocks share at most 4 points.
    # (0,1,2,3,4,5) and (0,1,2,3,4,6) share 0,1,2,3,4 (5 points).
    # This is INVALID for S(5,6,12).
    # They would cover the 5-subset (0,1,2,3,4) twice.
    
    # Correct second block choice:
    # Must not share 5 points with the first.
    # First: {0,1,2,3,4,5}
    # Second: {0,1,2,3,4,6} -> Shares 0,1,2,3,4. INVALID.
    # Second: {0,1,2,3,6,7} -> Shares 0,1,2,3. VALID.
    # Let's pick (0,1,2,3,6,7).
    
    second_block = (0, 1, 2, 3, 6, 7)
    second_block_idx = all_6_subsets.index(second_block)
    
    initial_mask = block_masks[first_block_idx] | block_masks[second_block_idx]
    initial_blocks = [list(first_block), list(second_block)]
    
    # Check for conflict (should be none by construction if we picked valid)
    if block_masks[first_block_idx] & block_masks[second_block_idx]:
        # Fallback to just first block if they conflict (they shouldn't if chosen correctly)
        initial_mask = block_masks[first_block_idx]
        initial_blocks = [list(first_block)]
        second_block_idx = -1

    def backtrack(mask, blocks):
        if time.time() > deadline:
            return None
            
        if mask == ALL_MASK:
            return blocks
            
        # Find an uncovered 5-subset
        uncovered_id = -1
        for id_ in range(792):
            if not (mask & (1 << id_)):
                uncovered_id = id_
                break
                
        if uncovered_id == -1:
            return blocks
            
        candidates = candidates_for_5[uncovered_id]
        
        for b_idx in candidates:
            # Skip if already used (mask check handles this if we track used blocks? 
            # No, mask check ensures no 5-subset overlap. 
            # If a block was used, its 5-subsets are in the mask. 
            # So any candidate sharing a 5-subset with a used block will have (b_mask & mask) != 0.
            # So we don't need a separate 'used' set for blocks, the mask suffices for validity.
            b_mask = block_masks[b_idx]
            if b_mask & mask:
                continue # Conflict
            
            new_mask = mask | b_mask
            new_blocks = blocks + [list(all_6_subsets[b_idx])]
            
            result = backtrack(new_mask, new_blocks)
            if result is not None:
                return result
                
        return None

    result = backtrack(initial_mask, initial_blocks)
    
    if result is not None:
        return result
    else:
        return initial_blocks
