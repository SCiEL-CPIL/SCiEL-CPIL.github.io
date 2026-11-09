import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) using an exact cover algorithm (Algorithm X)
    with bitset optimization. This finds the maximum set of 6-subsets of 0..11 such that
    no 5-subset is contained in more than one block. The optimum is 132 blocks.
    """
    START = time.time()
    TIME_LIMIT = 115.0  # Safety margin before 120s limit

    points = list(range(12))
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    n5 = len(all_5_subsets) # 792
    n6 = len(all_6_subsets) # 924

    # Map each 5-subset to an index
    five_index = {s5: i for i, s5 in enumerate(all_5_subsets)}
    
    # Precompute for each 6-subset, the bitmask of 5-subsets it covers
    # Using Python integers as bitsets for speed
    block_masks = []
    for b in all_6_subsets:
        mask = 0
        for s5 in combinations(b, 5):
            mask |= 1 << five_index[s5]
        block_masks.append(mask)
        
    # For each 5-subset, store the list of 6-subset indices that contain it
    five_to_blocks = [[] for _ in range(n5)]
    for bi, mask in enumerate(block_masks):
        # Iterate through set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            five_to_blocks[idx].append(bi)
            m ^= lsb
            
    # Algorithm X with bitmasks
    # State:
    # covered: bitmask of 5-subsets that are already covered by selected blocks
    # used_blocks: bitmask of 6-subsets that have been selected (to prevent reuse, though exact cover implies distinctness)
    # Actually, since we just need to maximize count and avoid 5-subset overlap, 
    # we don't strictly need to track used_blocks if we ensure we only pick blocks that don't conflict.
    # However, we must not pick the same block twice. 
    # A simpler state: 
    # 1. List of selected blocks
    # 2. Bitmask of covered 5-subsets
    # 3. Bitmask of available 6-subsets (initially all 1s)
    
    # To speed up, we can use a recursive search that picks the most constrained 5-subset
    # (the one with the fewest available covering blocks).
    
    # Precompute bitmasks for "which 5-subsets does block i cover" is block_masks[i].
    # We need to quickly find, for a given covered mask, which blocks are still valid.
    # A block is valid if (block_masks[bi] & covered) == 0.
    
    # Optimization: 
    # Instead of checking all 924 blocks at each step, we can maintain a list of candidate blocks.
    # But 924 is small. Let's just check validity efficiently.
    
    best_solution = []
    
    def search(covered_mask, available_blocks_mask, current_blocks):
        nonlocal best_solution
        if time.time() - START > TIME_LIMIT:
            return False
        
        # If we have a solution better than current best, update it
        # We are looking for the maximum number of blocks.
        # Theoretical max is 132. If we reach 132, we can stop.
        if len(current_blocks) == 132:
            best_solution = current_blocks[:]
            return True
            
        # Find the 5-subset with the minimum number of available covering blocks
        # We iterate through all 792 5-subsets.
        # For each uncovered 5-subset, count how many available blocks cover it.
        
        min_count = 1000
        min_block_idx = -1
        min_candidate_blocks = []
        
        # Optimization: We can prune if no uncovered 5-subsets remain? 
        # No, we want to maximize blocks. We stop when no more valid blocks can be added.
        
        # Check if any block can be added
        # We iterate through available blocks
        # But to choose the "best" branch (MRV heuristic), we look at constraints.
        
        # Let's iterate through available blocks to find one that can be added.
        # To apply MRV, we look for the 5-subset that is most constrained.
        
        # Efficient MRV:
        # For each 5-subset j not in covered_mask:
        #   count = number of available blocks that cover j and don't conflict with covered_mask
        #   if count == 0: return False (dead end, unless we've already maximized? No, if a 5-subset is uncovered and no block can cover it without conflict, we just can't cover it. But we are not required to cover ALL 5-subsets, just to maximize blocks.)
        
        # Wait, the problem is "Maximize the number of blocks".
        # Constraint: No 5-subset lies in two blocks.
        # This is a packing problem, not necessarily an exact cover of all 5-subsets.
        # However, the optimum 132 blocks cover exactly 132 * 6 = 792 5-subsets.
        # Since there are only 792 5-subsets total, an optimal solution MUST cover every 5-subset exactly once.
        # So it IS an exact cover problem of the 792 5-subsets by the 924 6-subsets (where each 6-subset covers 6 5-subsets).
        
        # So we can use Exact Cover logic.
        
        # MRV: Find the 5-subset with fewest available covering blocks.
        # If min_count == 0, we are in a dead end.
        
        # To speed up counting, we can precompute for each 5-subset the list of blocks.
        # Then for a given available_blocks_mask, we count how many are in the mask.
        
        # Let's try a simpler heuristic first: Just pick the first valid block.
        # Given the symmetry and uniqueness, this might work or might hit dead ends.
        # With 924 blocks, random or simple greedy might fail. 
        # Let's stick to MRV.
        
        # Iterate through 5-subsets
        for j in range(n5):
            if (covered_mask >> j) & 1:
                continue
                
            count = 0
            candidates = []
            for bi in five_to_blocks[j]:
                if (available_blocks_mask >> bi) & 1:
                    # Check if block bi conflicts with covered_mask
                    if (block_masks[bi] & covered_mask) == 0:
                        count += 1
                        candidates.append(bi)
                        
            if count < min_count:
                min_count = count
                min_candidate_blocks = candidates
                if count == 0:
                    return False
                if count == 1:
                    break # Can't get better than 1
            
        if min_count == 0:
            return False
            
        # Try each candidate block
        for bi in min_candidate_blocks:
            new_covered = covered_mask | block_masks[bi]
            new_available = available_blocks_mask & ~(1 << bi)
            new_blocks = current_blocks + [bi]
            
            if search(new_covered, new_available, new_blocks):
                return True
                
        return False

    initial_available = (1 << n6) - 1
    search(0, initial_available, [])
    
    if best_solution:
        return [list(all_6_subsets[bi]) for bi in best_solution]
    
    # Fallback if search fails (should not happen)
    # Return a greedy solution
    covered = 0
    result_blocks = []
    for bi in range(n6):
        if (block_masks[bi] & covered) == 0:
            result_blocks.append(all_6_subsets[bi])
            covered |= block_masks[bi]
    return result_blocks
