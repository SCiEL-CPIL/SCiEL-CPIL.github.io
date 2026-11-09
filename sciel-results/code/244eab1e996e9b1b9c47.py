import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    n = 12
    # Generate all 5-subsets and map to indices
    all_5_subsets = list(combinations(range(n), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    num_5_subsets = len(all_5_subsets)
    
    # Generate all 6-subsets (blocks) and their masks
    all_6_subsets = list(combinations(range(n), 6))
    blocks = []
    for b in all_6_subsets:
        mask = 0
        for i in range(len(b)):
            # Get 5-subset by removing b[i]
            s = tuple(x for j, x in enumerate(b) if j != i)
            idx = subset_to_idx[s]
            mask |= (1 << idx)
        blocks.append((b, mask))
    
    # Precompute for each 5-subset index, which blocks cover it
    # This helps in finding candidates quickly
    # But since we use "most constrained", we need to check candidates for a specific 5-subset
    # Let's invert: for each 5-subset index, list of block indices that contain it
    subset_to_blocks = [[] for _ in range(num_5_subsets)]
    for i, (b, mask) in enumerate(blocks):
        # Extract indices from mask
        # Since mask has exactly 6 bits set, we can iterate
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            subset_to_blocks[idx].append(i)
            m ^= lsb

    best_solution = []
    
    # Use a recursive backtracker
    # To avoid deep recursion issues, we can use a stack, but Python recursion limit is usually high enough for 132 depth
    # However, the branching factor is small due to constraints.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def solve(covered_mask, current_blocks, block_indices_used):
        if time.time() - start_time > timeout:
            return False
            
        if len(current_blocks) == 132:
            best_solution.extend(current_blocks)
            return True
            
        # Find an uncovered 5-subset with the fewest available covering blocks
        min_candidates = None
        min_count = float('inf')
        target_subset_idx = -1
        
        # Iterate through all 5-subsets to find an uncovered one
        # Optimization: We can just pick the first uncovered one with fewest options
        for s_idx in range(num_5_subsets):
            if (covered_mask >> s_idx) & 1:
                continue # Already covered
                
            # Check which blocks covering s_idx are still valid (disjoint from covered_mask)
            valid_blocks = []
            for b_idx in subset_to_blocks[s_idx]:
                if b_idx in block_indices_used:
                    continue
                b_mask = blocks[b_idx][1]
                if b_mask & covered_mask:
                    continue
                valid_blocks.append(b_idx)
                
            count = len(valid_blocks)
            if count < min_count:
                min_count = count
                target_subset_idx = s_idx
                min_candidates = valid_blocks
                if count == 0:
                    return False # Dead end
                if count == 1:
                    break # Can't get better than 1
                    
        if min_candidates is None:
            # All subsets covered? But we only have 132 blocks max.
            # If we haven't reached 132, we failed.
            return False
            
        for b_idx in min_candidates:
            b, b_mask = blocks[b_idx]
            new_covered = covered_mask | b_mask
            new_used = block_indices_used | {b_idx}
            
            if solve(new_covered, current_blocks + [b], new_used):
                return True
                
        return False

    # Start search
    # To break symmetry and speed up, we can fix the first block.
    # The first block in lex order is (0,1,2,3,4,5).
    # Let's just run the generic solver. It should be fast.
    
    if solve(0, [], set()):
        return best_solution
    else:
        # Fallback: return whatever we have? Or empty?
        # The task says maximize. If we fail, we return best found.
        # But we expect success.
        return []
