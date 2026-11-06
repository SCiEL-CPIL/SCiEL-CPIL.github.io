import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    points = range(12)
    
    # 1. Enumerate all 5-subsets
    all_5_subsets = list(combinations(points, 5))
    n_5 = len(all_5_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # 2. Enumerate all 6-subsets
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks)
    
    # 3. Compute masks
    block_masks = [0] * n_blocks
    blocks_covering = [[] for _ in range(n_5)]
    
    for i, b in enumerate(all_blocks):
        mask = 0
        for s in combinations(b, 5):
            idx = subset_to_idx[s]
            mask |= (1 << idx)
            blocks_covering[idx].append(i)
        block_masks[i] = mask
        
    # Symmetry breaking: Fix first block
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_blocks.index(first_block)
    
    initial_uncovered = (1 << n_5) - 1
    initial_uncovered &= ~block_masks[first_block_idx]
    
    selected_blocks = [first_block]
    
    def solve(uncovered_mask, selected_count):
        if time.time() > deadline:
            return None
            
        if uncovered_mask == 0:
            return selected_blocks[:]
            
        # MRV: Find 5-subset with min candidates
        min_count = float('inf')
        best_subset_idx = -1
        best_candidates = []
        
        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb
            
            count = 0
            candidates = []
            for b_idx in blocks_covering[idx]:
                if block_masks[b_idx] & ~uncovered_mask == 0:
                    count += 1
                    candidates.append(b_idx)
                    
            if count < min_count:
                min_count = count
                best_subset_idx = idx
                best_candidates = candidates
                
            if min_count == 0:
                return None
                
        # Backtrack
        for b_idx in best_candidates:
            new_uncovered = uncovered_mask & ~block_masks[b_idx]
            selected_blocks.append(all_blocks[b_idx])
            
            result = solve(new_uncovered, selected_count + 1)
            if result is not None:
                return result
                
            selected_blocks.pop()
            
        return None

    result = solve(initial_uncovered, 1)
    
    if result is None:
        return selected_blocks[:]
        
    return result
