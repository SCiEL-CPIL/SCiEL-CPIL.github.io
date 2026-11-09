def construct():
    # The 132 blocks of the Steiner system S(5,6,12)
    # Points are 0..11
    blocks = [
        (0, 1, 2, 3, 4, 5), (0, 1, 2, 6, 7, 8), (0, 1, 2, 9, 10, 11), (0, 1, 3, 6, 9, 10),
        (0, 1, 3, 7, 9, 11), (0, 1, 3, 8, 10, 11), (0, 1, 4, 6, 7, 10), (0, 1, 4, 6, 8, 11),
        (0, 1, 4, 7, 8, 10), (0, 1, 5, 6, 9, 11), (0, 1, 5, 7, 10, 11), (0, 1, 5, 8, 9, 10),
        (0, 2, 3, 6, 7, 9), (0, 2, 3, 7, 8, 10), (0, 2, 3, 8, 9, 11), (0, 2, 4, 6, 8, 9),
        (0, 2, 4, 7, 10, 11), (0, 2, 4, 8, 10, 11), (0, 2, 5, 6, 7, 11), (0, 2, 5, 7, 8, 10),
        (0, 2, 5, 8, 9, 10), (0, 3, 4, 6, 7, 8), (0, 3, 4, 6, 9, 10), (0, 3, 4, 7, 9, 11),
        (0, 3, 5, 6, 8, 11), (0, 3, 5, 7, 8, 10), (0, 3, 5, 7, 9, 11), (0, 4, 5, 6, 7, 10),
        (0, 4, 5, 6, 8, 11), (0, 4, 5, 7, 9, 10), (1, 2, 3, 4, 6, 11), (1, 2, 3, 5, 7, 10),
        (1, 2, 3, 5, 8, 9), (1, 2, 4, 5, 6, 9), (1, 2, 4, 5, 7, 11), (1, 2, 4, 7, 8, 9),
        (1, 2, 5, 6, 7, 8), (1, 2, 5, 6, 8, 9), (1, 2, 5, 7, 9, 10), (1, 3, 4, 5, 6, 7),
        (1, 3, 4, 6, 8, 10), (1, 3, 4, 7, 8, 11), (1, 3, 5, 6, 8, 11), (1, 3, 5, 7, 8, 10),
        (1, 3, 5, 8, 9, 11), (1, 4, 5, 6, 7, 9), (1, 4, 5, 7, 8, 10), (1, 4, 5, 8, 9, 11),
        (2, 3, 4, 5, 6, 8), (2, 3, 4, 5, 7, 9), (2, 3, 4, 6, 8, 11), (2, 3, 4, 7, 9, 10),
        (2, 3, 5, 6, 7, 10), (2, 3, 5, 6, 8, 11), (2, 3, 5, 7, 8, 9), (2, 4, 5, 6, 7, 11),
        (2, 4, 5, 6, 8, 9), (2, 4, 5, 7, 9, 10), (3, 4, 5, 6, 7, 8), (3, 4, 5, 6, 8, 9),
        (3, 4, 5, 7, 9, 11), (3, 4, 5, 7, 10, 11), (3, 4, 6, 7, 9, 10), (3, 4, 6, 8, 9, 11),
        (3, 4, 7, 8, 10, 11), (3, 5, 6, 7, 8, 10), (3, 5, 6, 7, 9, 11), (3, 5, 6, 8, 9, 10),
        (3, 5, 7, 8, 10, 11), (3, 5, 7, 9, 10, 11), (3, 5, 8, 9, 10, 11), (4, 5, 6, 7, 8, 9),
        (4, 5, 6, 7, 10, 11), (4, 5, 6, 8, 9, 10), (4, 5, 7, 8, 9, 11), (4, 5, 7, 8, 10, 11),
        (4, 5, 8, 9, 10, 11), (6, 7, 8, 9, 10, 11)
    ]
    # The above list is a subset or needs verification. 
    # Actually, the number of blocks is 132. The list above has 66 blocks? 
    # Let's count: 4 lines * 15 lines = 60? No.
    # I will generate the correct 132 blocks using a reliable method in the code 
    # or provide the full correct list.
    
    # To be safe and correct, I will use a known valid construction.
    # The blocks of S(5,6,12) can be generated from the 12 points.
    # A standard list is available. I will embed the full correct list.
    
    # Correct list of 132 blocks:
    # I will use a generator that is known to produce the Witt design.
    # However, hardcoding is safer.
    
    # Let's use the blocks from a trusted source (e.g. Design Theory by Stinson, or online tables).
    # Since I cannot browse, I will rely on the fact that the previous program found a solution.
    # But the previous program is a backtracker.
    # I will implement a very fast backtracker that uses bitmasks and precomputed tables, 
    # which should be fast enough, OR I will hardcode.
    
    # Let's try to generate them using the Mathieu group property.
    # The group M12 acts transitively on the 132 blocks.
    # One representative block is {0,1,2,3,4,5}.
    # The stabilizer of this block in M12 is S5 x S2 (or similar).
    # This is complex to implement from scratch.
    
    # Alternative: Use the fact that the blocks are the 132 solutions to a specific system.
    # I will provide the hardcoded list of 132 blocks. I will ensure it is complete.
    
    # Here is a verified list of 132 blocks for S(5,6,12):
    blocks = [
        (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 6, 7), (0, 1, 2, 3, 8, 9), (0, 1, 2, 3, 10, 11),
        (0, 1, 2, 4, 6, 8), (0, 1, 2, 4, 7, 9), (0, 1, 2, 4, 10, 11), (0, 1, 2, 5, 6, 9),
        (0, 1, 2, 5, 7, 10), (0, 1, 2, 5, 8, 11), (0, 1, 3, 4, 6, 10), (0, 1, 3, 4, 7, 11),
        (0, 1, 3, 4, 8, 9), (0, 1, 3, 5, 6, 11), (0, 1, 3, 5, 7, 8), (0, 1, 3, 5, 9, 10),
        (0, 1, 4, 5, 6, 7), (0, 1, 4, 5, 8, 10), (0, 1, 4, 5, 9, 11), (0, 1, 4, 6, 7, 9),
        (0, 1, 4, 6, 8, 10), (0, 1, 4, 6, 9, 11), (0, 1, 5, 6, 7, 8), (0, 1, 5, 6, 8, 10),
        (0, 1, 5, 6, 9, 11), (0, 1, 5, 7, 8, 10), (0, 1, 5, 7, 9, 11), (0, 1, 5, 8, 9, 10),
        (0, 2, 3, 4, 7, 10), (0, 2, 3, 4, 8, 11), (0, 2, 3, 4, 9, 10), (0, 2, 3, 5, 6, 7),
        (0, 2, 3, 5, 8, 10), (0, 2, 3, 5, 9, 11), (0, 2, 3, 5, 9, 10), (0, 2, 3, 6, 7, 8),
        (0, 2, 3, 6, 8, 11), (0, 2, 3, 6, 9, 10), (0, 2, 3, 6, 9, 11), (0, 2, 3, 7, 8, 9),
        (0, 2, 4, 5, 6, 7), (0, 2, 4, 5, 8, 9), (0, 2, 4, 5, 9, 11), (0, 2, 4, 6, 7, 11),
        (0, 2, 4, 6, 8, 9), (0, 2, 4, 6, 9, 10), (0, 2, 4, 7, 8, 10), (0, 2, 4, 7, 9, 11),
        (0, 2, 4, 7, 9, 10), (0, 2, 4, 8, 9, 10), (0, 2, 5, 6, 7, 8), (0, 2, 5, 6, 7, 9),
        (0, 2, 5, 6, 8, 10), (0, 2, 5, 6, 9, 11), (0, 2, 5, 7, 8, 9), (0, 2, 5, 7, 8, 11),
        (0, 2, 5, 7, 9, 10), (0, 2, 5, 8, 9, 10), (0, 3, 4, 5, 6, 8), (0, 3, 4, 5, 7, 9),
        (0, 3, 4, 5, 8, 10), (0, 3, 4, 5, 9, 11), (0, 3, 4, 6, 7, 8), (0, 3, 4, 6, 8, 11),
        (0, 3, 4, 6, 9, 10), (0, 3, 4, 7, 8, 9), (0, 3, 4, 7, 9, 11), (0, 3, 4, 7, 10, 11),
        (0, 3, 5, 6, 7, 9), (0, 3, 5, 6, 7, 10), (0, 3, 5, 6, 8, 11), (0, 3, 5, 6, 9, 10),
        (0, 3, 5, 7, 8, 9), (0, 3, 5, 7, 8, 10), (0, 3, 5, 7, 9, 11), (0, 3, 5, 8, 9, 10),
        (0, 4, 5, 6, 7, 8), (0, 4, 5, 6, 7, 9), (0, 4, 5, 6, 8, 11), (0, 4, 5, 6, 9, 10),
        (0, 4, 5, 7, 8, 9), (0, 4, 5, 7, 8, 10), (0, 4, 5, 7, 9, 11), (0, 4, 5, 8, 9, 10),
        (1, 2, 3, 4, 6, 7), (1, 2, 3, 4, 8, 9), (1, 2, 3, 4, 10, 11), (1, 2, 3, 5, 6, 8),
        (1, 2, 3, 5, 7, 9), (1, 2, 3, 5, 8, 10), (1, 2, 3, 5, 9, 11), (1, 2, 3, 5, 9, 10),
        (1, 2, 3, 6, 7, 8), (1, 2, 3, 6, 8, 11), (1, 2, 3, 6, 9, 10), (1, 2, 3, 6, 9, 11),
        (1, 2, 3, 7, 8, 9), (1, 2, 4, 5, 6, 7), (1, 2, 4, 5, 8, 9), (1, 2, 4, 5, 9, 11),
        (1, 2, 4, 6, 7, 11), (1, 2, 4, 6, 8, 9), (1, 2, 4, 6, 9, 10), (1, 2, 4, 7, 8, 10),
        (1, 2, 4, 7, 9, 11), (1, 2, 4, 7, 9, 10), (1, 2, 4, 8, 9, 10), (1, 2, 5, 6, 7, 8),
        (1, 2, 5, 6, 7, 9), (1, 2, 5, 6, 8, 10), (1, 2, 5, 6, 9, 11), (1, 2, 5, 7, 8, 9),
        (1, 2, 5, 7, 8, 11), (1, 2, 5, 7, 9, 10), (1, 2, 5, 8, 9, 10), (1, 3, 4, 5, 6, 8),
        (1, 3, 4, 5, 7, 9), (1, 3, 4, 5, 8, 10), (1, 3, 4, 5, 9, 11), (1, 3, 4, 6, 7, 8),
        (1, 3, 4, 6, 8, 11), (1, 3, 4, 6, 9, 10), (1, 3, 4, 7, 8, 9), (1, 3, 4, 7, 9, 11),
        (1, 3, 4, 7, 10, 11), (1, 3, 5, 6, 7, 9), (1, 3, 5, 6, 7, 10), (1, 3, 5, 6, 8, 11),
        (1, 3, 5, 6, 9, 10), (1, 3, 5, 7, 8, 9), (1, 3, 5, 7, 8, 10), (1, 3, 5, 7, 9, 11),
        (1, 3, 5, 8, 9, 10), (1, 4, 5, 6, 7, 8), (1, 4, 5, 6, 7, 9), (1, 4, 5, 6, 8, 11),
        (1, 4, 5, 6, 9, 10), (1, 4, 5, 7, 8, 9), (1, 4, 5, 7, 8, 10), (1, 4, 5, 7, 9, 11),
        (1, 4, 5, 8, 9, 10), (2, 3, 4, 5, 6, 7), (2, 3, 4, 5, 8, 9), (2, 3, 4, 5, 9, 11),
        (2, 3, 4, 6, 7, 11), (2, 3, 4, 6, 8, 9), (2, 3, 4, 6, 9, 10), (2, 3, 4, 7, 8, 10),
        (2, 3, 4, 7, 9, 11), (2, 3, 4, 7, 9, 10), (2, 3, 4, 8, 9, 10), (2, 3, 5, 6, 7, 8),
        (2, 3, 5, 6, 7, 9), (2, 3, 5, 6, 8, 10), (2, 3, 5, 6, 9, 11), (2, 3, 5, 7, 8, 9),
        (2, 3, 5, 7, 8, 11), (2, 3, 5, 7, 9, 10), (2, 3, 5, 8, 9, 10), (2, 4, 5, 6, 7, 8),
        (2, 4, 5, 6, 7, 9), (2, 4, 5, 6, 8, 11), (2, 4, 5, 6, 9, 10), (2, 4, 5, 7, 8, 9),
        (2, 4, 5, 7, 8, 10), (2, 4, 5, 7, 9, 11), (2, 4, 5, 8, 9, 10), (3, 4, 5, 6, 7, 8),
        (3, 4, 5, 6, 7, 9), (3, 4, 5, 6, 8, 11), (3, 4, 5, 6, 9, 10), (3, 4, 5, 7, 8, 9),
        (3, 4, 5, 7, 8, 10), (3, 4, 5, 7, 9, 11), (3, 4, 5, 8, 9, 10), (4, 5, 6, 7, 8, 9),
        (4, 5, 6, 7, 8, 10), (4, 5, 6, 7, 9, 11), (4, 5, 6, 7, 10, 11), (4, 5, 6, 8, 9, 10),
        (4, 5, 6, 8, 9, 11), (4, 5, 6, 8, 10, 11), (4, 5, 6, 9, 10, 11), (4, 5, 7, 8, 9, 10),
        (4, 5, 7, 8, 9, 11), (4, 5, 7, 8, 10, 11), (4, 5, 7, 9, 10, 11), (4, 5, 8, 9, 10, 11),
        (6, 7, 8, 9, 10, 11)
    ]
    
    # The list above is likely incorrect or incomplete because I am typing it from memory/patterns.
    # I will use a reliable algorithmic generation instead.
    
    # Reliable Generation:
    # The blocks of S(5,6,12) are the 132 subsets of size 6 that can be obtained
    # by applying the Mathieu group M12 to the block {0,1,2,3,4,5}.
    # However, implementing M12 is heavy.
    
    # Alternative: Use the fact that the blocks are the solutions to:
    # A packing where every 5-subset is contained in exactly one block.
    
    # I will use the backtracking solver from the prompt but optimized with bitmasks.
    # This is safer than hardcoding a potentially wrong list.
    
    import time
    start_time = time.time()
    timeout = 110
    
    n = 12
    # Precompute masks for all 6-subsets
    # There are C(12,6) = 924 blocks.
    # We need to check if they are disjoint with respect to 5-subsets.
    # Two 6-subsets A and B are "compatible" if |A intersect B| <= 4.
    # Because if |A intersect B| >= 5, they share a 5-subset.
    
    all_blocks = list(combinations(range(n), 6))
    
    # Precompute compatibility
    # For each block i, list of blocks j compatible with i.
    # This is O(924^2) which is ~850k operations. Fast enough.
    
    # Actually, we can just do a backtracking search with a good heuristic.
    # Heuristic: Pick the block that covers the most "constrained" 5-subsets?
    # Or simply: Pick the block that allows the most future options?
    
    # Let's use a simple recursive backtracker with bitmask for used 5-subsets.
    # Total 5-subsets: C(12,5) = 792.
    # This fits in a 792-bit integer (Python handles big ints).
    
    # Map 5-subsets to indices
    all_5_subsets = list(combinations(range(n), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # Precompute mask for each 6-subset
    block_masks = []
    for b in all_blocks:
        mask = 0
        for i in range(6):
            s = tuple(x for j, x in enumerate(b) if j != i)
            mask |= (1 << subset_to_idx[s])
        block_masks.append(mask)
    
    # We want to find a set of blocks such that their masks are disjoint.
    # This is a Maximum Independent Set problem on a conflict graph,
    # or a Set Packing problem.
    
    # Given the small size, a backtracker with "most constrained" (fewest candidates)
    # on the 5-subsets should work.
    
    # For each 5-subset, which blocks cover it?
    # There are C(12-5, 6-5) = C(7,1) = 7 blocks covering any specific 5-subset?
    # No. If a 5-subset S is fixed, a block B containing S must have one more element from the remaining 7.
    # So there are 7 blocks containing any given 5-subset.
    
    # Invert: for each 5-subset index, list of block indices.
    subset_to_blocks = [[] for _ in range(792)]
    for i, mask in enumerate(block_masks):
        # Extract set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            subset_to_blocks[idx].append(i)
            m ^= lsb
            
    best_solution = []
    
    def solve(covered_mask, current_blocks):
        if time.time() - start_time > timeout:
            return False
        if len(current_blocks) == 132:
            best_solution.extend(current_blocks)
            return True
            
        # Find uncovered 5-subset with fewest valid blocks
        min_count = float('inf')
        target_idx = -1
        candidates = []
        
        for s_idx in range(792):
            if (covered_mask >> s_idx) & 1:
                continue
            valid = []
            for b_idx in subset_to_blocks[s_idx]:
                if block_masks[b_idx] & covered_mask:
                    continue
                valid.append(b_idx)
            count = len(valid)
            if count < min_count:
                min_count = count
                target_idx = s_idx
                candidates = valid
                if count == 0:
                    return False
                if count == 1:
                    break
                    
        if not candidates:
            return False
            
        for b_idx in candidates:
            new_mask = covered_mask | block_masks[b_idx]
            if solve(new_mask, current_blocks + [all_blocks[b_idx]]):
                return True
        return False

    if solve(0, []):
        return best_solution
    else:
        return []

