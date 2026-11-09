import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    points = range(12)
    
    # Map 5-subsets to indices
    all_5_subs = list(combinations(points, 5))
    sub_to_idx = {s: i for i, s in enumerate(all_5_subs)}
    num_5_subs = len(all_5_subs) # 792
    
    # Generate all 6-subsets (blocks) and their masks
    all_6_subs = list(combinations(points, 6))
    block_masks = []
    for b in all_6_subs:
        mask = 0
        # A 6-subset contains 6 5-subsets (remove each element)
        for i in range(6):
            subset = tuple(x for j, x in enumerate(b) if j != i)
            # Ensure subset is sorted for lookup (combinations are sorted)
            # b is from combinations, so sorted. Removing one keeps order.
            idx = sub_to_idx[subset]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(block_masks) # 924
    
    # Precompute conflicts: for each block, which other blocks conflict with it?
    # Conflict means they share at least one 5-subset.
    # Since we are looking for a packing, we just need to ensure disjointness.
    # We can check disjointness on the fly: mask1 & mask2 == 0.
    
    # Algorithm X style search
    # We want to find a set of blocks such that their union of masks is all 1s (exact cover)
    # Because 132 * 6 = 792, any valid packing of size 132 is an exact cover.
    
    # State:
    # covered_mask: bits of 5-subsets already covered
    # available: list of indices of blocks still available (disjoint from chosen ones)
    # current_solution: list of block indices
    
    # To speed up, we can use a heuristic: pick the 5-subset with fewest available covering blocks.
    # We need to maintain counts of how many available blocks cover each 5-subset.
    
    # Initial counts
    # col_count[c] = number of blocks in 'available' that cover c
    # This is expensive to update if we do it naively.
    
    # Alternative: Just use a recursive backtracking with a good ordering.
    # Order blocks? Or pick column dynamically.
    
    # Let's use a standard DLX-like approach with bitsets.
    
    # We represent the set of available blocks as a bitmask of 924 bits?
    # 924 bits is about 116 bytes. Python int handles this.
    # Let's use a list of available block indices for simplicity, but filter them.
    
    # Actually, a simpler approach for this specific small problem:
    # The Witt design is unique up to isomorphism.
    # We can just run a backtracking search that stops when it finds 132 blocks.
    
    # Heuristic: 
    # 1. Pick a 5-subset 'c' that is uncovered.
    # 2. Among available blocks covering 'c', pick one.
    # 3. Remove all conflicting blocks.
    # 4. Recurse.
    
    # To pick 'c' efficiently, we can just pick the first uncovered 5-subset.
    # This might be slow if it leads to dead ends.
    # Better: Pick the uncovered 5-subset with the fewest available covering blocks.
    
    # Let's implement this.
    
    def solve(covered_mask, available_blocks, solution):
        if time.time() > deadline:
            return None
        
        if len(solution) == 132:
            # Check if it's a valid exact cover (should be if we only add disjoint blocks)
            # Sum of bits should be 792.
            # Since we started with 0 and added 6 disjoint bits each time, 
            # 132 * 6 = 792 bits. If they are disjoint, it's a cover.
            return solution[:]
            
        if not available_blocks:
            return None
            
        # Find the uncovered 5-subset with the minimum number of available covering blocks
        min_count = float('inf')
        target_col = -1
        
        # We need to know which 5-subsets are uncovered.
        # uncovered_mask = (~covered_mask) & ((1 << num_5_subs) - 1)
        # Iterating through all 792 subsets to find min count is O(792 * N_avail)
        # N_avail starts at 924. 792 * 924 ~ 700k ops per step.
        # Depth 132. 132 * 700k ~ 90M ops. Might be too slow in Python?
        
        # Optimization:
        # Maintain counts dynamically.
        
        # Let's try a simpler heuristic first: 
        # Just pick the first available block? No, greedy fails.
        # Pick the block that covers the most "rare" 5-subsets?
        
        # Let's stick to the dynamic count if possible, or a simpler check.
        
        # Actually, for 12 points, a simple backtracking with "pick first uncovered column" 
        # and "try blocks in order" might work if the order is good.
        # But "min count" is the standard DLX heuristic.
        
        # Let's implement min count selection.
        # To avoid recalculating counts for all columns every time, 
        # we can just calculate it for the uncovered ones.
        
        all_mask = (1 << num_5_subs) - 1
        uncovered_mask = all_mask ^ covered_mask
        
        # Iterate through set bits in uncovered_mask
        # For each such bit c, count how many blocks in available_blocks have bit c set.
        
        # This is still O(N_avail * num_uncovered).
        
        # Given the small size, let's just do it.
        
        # Optimization: Precompute for each 5-subset c, the list of blocks that cover it.
        # col_to_blocks[c] = [b1, b2, ...]
        # Then for a given c, count = len([b for b in available_blocks if b in col_to_blocks[c]])
        # This is O(len(col_to_blocks[c])).
        # Average len is 792*6/924 = 5.14.
        # So counting is very fast.
        
        # Precompute col_to_blocks
        col_to_blocks = [[] for _ in range(num_5_subs)]
        for i, mask in enumerate(block_masks):
            # Iterate bits in mask
            m = mask
            while m:
                lsb = m & -m
                c = lsb.bit_length() - 1
                col_to_blocks[c].append(i)
                m ^= lsb
                
        # Now in solve:
        min_count = float('inf')
        target_col = -1
        
        # Iterate uncovered columns
        m = uncovered_mask
        while m:
            lsb = m & -m
            c = lsb.bit_length() - 1
            # Count available blocks covering c
            # available_blocks is a list. 
            # Checking membership in col_to_blocks[c] (list) is O(N_avail * len(col_to_blocks[c]))?
            # No, we can just check if block is in available set.
            # Let's keep available_blocks as a set for O(1) lookup?
            # Or a boolean array?
            
            # Let's pass available as a set of indices.
            
            count = 0
            for b in col_to_blocks[c]:
                if b in available_blocks:
                    count += 1
                    if count >= min_count:
                        break # Pruning: no need to count more if we already have min
            if count < min_count:
                min_count = count
                target_col = c
                if min_count == 1:
                    break # Can't get better than 1
            m ^= lsb
            
        if target_col == -1:
            return None # All covered? But len(solution) < 132. 
            # If all covered, we have a cover. If len(solution) != 132, it's not exact?
            # But 6*len = 792 implies len=132.
            
        # Get candidate blocks for target_col
        candidates = [b for b in col_to_blocks[target_col] if b in available_blocks]
        
        # Order candidates? Maybe by how many other conflicts they cause?
        # For now, just try them.
        
        for b in candidates:
            # Check if b is still available (it is)
            # New available: remove b and all blocks conflicting with b
            # Conflict if (block_masks[b] & block_masks[other]) != 0
            
            new_available = set()
            for other in available_blocks:
                if other == b:
                    continue
                if block_masks[b] & block_masks[other]:
                    continue
                new_available.add(other)
                
            new_covered = covered_mask | block_masks[b]
            
            res = solve(new_covered, new_available, solution + [b])
            if res is not None:
                return res
                
        return None

    # Initial call
    initial_available = set(range(num_blocks))
    initial_covered = 0
    res = solve(initial_covered, initial_available, [])
    
    if res is None:
        # Fallback: Return empty or a greedy result if search fails (shouldn't happen)
        # But for safety, return a small valid packing
        # Greedy
        covered = 0
        blocks = []
        for i in range(num_blocks):
            if block_masks[i] & covered:
                continue
            blocks.append(list(all_6_subs[i]))
            covered |= block_masks[i]
        return blocks
        
    # Convert indices to lists of points
    return [list(all_6_subs[i]) for i in res]
