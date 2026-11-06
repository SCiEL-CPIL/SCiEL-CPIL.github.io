import itertools
import time
import sys

def construct():
    points = range(12)
    all_5sets = list(itertools.combinations(points, 5))
    n_5sets = len(all_5sets)
    five_set_index = {s: i for i, s in enumerate(all_5sets)}
    
    all_6sets = list(itertools.combinations(points, 6))
    n_6sets = len(all_6sets)
    
    # For each block, compute the bitmask of 5-subsets it covers
    # And for each 5-set, list the blocks that cover it
    block_masks = [0] * n_6sets
    blocks_by_5set = [[] for _ in range(n_5sets)]
    
    for i, block in enumerate(all_6sets):
        mask = 0
        for j in range(6):
            subset = tuple(block[:j] + block[j+1:])
            idx = five_set_index[subset]
            mask |= (1 << idx)
            blocks_by_5set[idx].append(i)
        block_masks[i] = mask
    
    # Precompute for each block, the list of 5-set indices it covers
    block_5sets = []
    for i in range(n_6sets):
        indices = []
        mask = block_masks[i]
        while mask:
            lb = mask & -mask
            idx = lb.bit_length() - 1
            indices.append(idx)
            mask ^= lb
        block_5sets.append(indices)
    
    best_blocks = []
    target_size = 132
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute tuple-to-index mapping for fast lookup
    tuple_to_idx = {t: i for i, t in enumerate(all_6sets)}
    
    def backtrack(covered_mask, current_blocks):
        nonlocal best_blocks
        if len(current_blocks) == target_size:
            best_blocks = list(current_blocks)
            return True
            
        if time.time() > deadline:
            return False
            
        available = (~covered_mask) & ((1 << n_5sets) - 1)
        if available == 0:
            return len(current_blocks) == target_size

        # MRV: Find the uncovered 5-set with minimum number of compatible blocks
        best_idx = -1
        best_count = float('inf')
        best_candidates = []
        
        temp = available
        while temp:
            lb = temp & -temp
            idx_5set = lb.bit_length() - 1
            temp ^= lb
            
            candidates = blocks_by_5set[idx_5set]
            count = 0
            for b_idx in candidates:
                if not (block_masks[b_idx] & covered_mask):
                    count += 1
            if count < best_count:
                best_count = count
                best_idx = idx_5set
                best_candidates = [b_idx for b_idx in candidates if not (block_masks[b_idx] & covered_mask)]
                if best_count == 1:
                    break
        
        if best_count == 0:
            return False
            
        # Sort candidates: prefer blocks whose 5-sets have fewer alternative blocks
        def candidate_score(b_idx):
            score = 0
            for idx_5set in block_5sets[b_idx]:
                count = 0
                for other_b in blocks_by_5set[idx_5set]:
                    if not (block_masks[other_b] & covered_mask):
                        count += 1
                score += count
            return score
        
        best_candidates.sort(key=candidate_score)
        
        for b_idx in best_candidates:
            mask = block_masks[b_idx]
            current_blocks.append(b_idx)
            if backtrack(covered_mask | mask, current_blocks):
                return True
            current_blocks.pop()
            
        return False

    # Symmetry breaking: The first block must contain {0, 1, 2, 3, 4}.
    for x in range(5, 12):
        first_block_tuple = (0, 1, 2, 3, 4, x)
        first_block_idx = tuple_to_idx[first_block_tuple]
        
        initial_mask = block_masks[first_block_idx]
        current_blocks = [first_block_idx]
        
        if backtrack(initial_mask, current_blocks):
            result = [list(all_6sets[i]) for i in best_blocks]
            return result
            
    if best_blocks:
        return [list(all_6sets[i]) for i in best_blocks]
    return []
