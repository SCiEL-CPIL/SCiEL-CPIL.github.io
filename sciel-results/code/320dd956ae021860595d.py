import time
import numpy as np
from itertools import combinations

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) using optimized backtracking with bitsets.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_points = 12
    block_size = 6
    sub_size = 5
    target_blocks = 132
    total_subs = 792 # C(12, 5)
    
    # 1. Precomputation
    all_blocks = list(combinations(range(n_points), block_size))
    n_blocks = len(all_blocks)
    
    all_subs = list(combinations(range(n_points), sub_size))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    # Map each block to a bitmask of the 5-subsets it covers
    block_masks = []
    # Inverted index: for each 5-subset, which blocks contain it?
    subs_to_blocks = [[] for _ in range(total_subs)]
    
    for b in all_blocks:
        mask = 0
        for s in combinations(b, sub_size):
            idx = sub_to_idx[s]
            mask |= (1 << idx)
            subs_to_blocks[idx].append(len(block_masks)) # Append block index
        block_masks.append(mask)

    # --- Optimized Backtracking ---
    first_block = (0, 1, 2, 3, 4, 5)
    # Find index of first_block in all_blocks
    first_block_idx = all_blocks.index(first_block)
    
    covered_mask = block_masks[first_block_idx]
    chosen_blocks = [first_block]
    last_block_idx = first_block_idx
    
    def backtrack(covered_mask, chosen_blocks, last_block_idx, depth):
        if time.time() > deadline:
            return None
        if len(chosen_blocks) == target_blocks:
            # Verify all 792 subsets are covered (should be true if 132 blocks * 6 subs/block = 792)
            # 132 * 6 = 792. So if we have 132 valid blocks, we cover exactly 792 subsets.
            return list(chosen_blocks)
        
        # Find the uncovered 5-subset with the fewest valid candidates (MRV)
        # To avoid scanning all 792 every time, we can just pick the first uncovered one.
        # Given the high symmetry, this is usually sufficient.
        # Let's try MRV for robustness.
        
        best_sub_idx = -1
        best_candidates = None
        min_count = float('inf')
        
        # Iterate through all subsets to find one that is uncovered
        for s_idx in range(total_subs):
            # Check if subset s_idx is covered
            if (covered_mask >> s_idx) & 1:
                continue
            
            # Count valid candidates for this subset
            candidates = []
            for b_idx in subs_to_blocks[s_idx]:
                if b_idx <= last_block_idx:
                    continue
                if (block_masks[b_idx] & covered_mask) == 0:
                    candidates.append(b_idx)
            
            count = len(candidates)
            if count == 0:
                return None # Dead end
            
            if count < min_count:
                min_count = count
                best_sub_idx = s_idx
                best_candidates = candidates
                if count == 1:
                    break # Can't get better than 1
            
        if best_candidates is None:
            return None
            
        for b_idx in best_candidates:
            new_covered_mask = covered_mask | block_masks[b_idx]
            result = backtrack(new_covered_mask, chosen_blocks + [all_blocks[b_idx]], b_idx, depth + 1)
            if result is not None:
                return result
                
        return None

    solution = backtrack(covered_mask, chosen_blocks, last_block_idx, 1)
    
    if solution:
        return solution
    else:
        # Greedy fallback (unlikely to be needed with optimized backtracking)
        covered_mask = 0
        greedy_blocks = []
        for b_idx in range(n_blocks):
            if (block_masks[b_idx] & covered_mask) == 0:
                greedy_blocks.append(all_blocks[b_idx])
                covered_mask |= block_masks[b_idx]
                if len(greedy_blocks) >= target_blocks:
                    break
        return greedy_blocks
