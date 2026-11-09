import itertools
import time

def construct():
    """
    Construct the Witt design S(5,6,12) with 132 blocks using a
    bitmask-based Exact Cover (Algorithm X) solver.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    points = list(range(12))
    
    # Generate all 6-subsets (blocks)
    all_blocks = list(itertools.combinations(points, 6))
    
    # Map 5-subsets to indices 0..791
    all_5subsets = list(itertools.combinations(points, 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    num_cols = len(all_5subsets) # 792
    ALL_ONES = (1 << num_cols) - 1
    
    # Precompute bitmask for each block
    # mask[b] has bits set for the 5-subsets contained in b
    block_masks = []
    for b in all_blocks:
        mask = 0
        for s in itertools.combinations(b, 5):
            mask |= (1 << subset_to_idx[s])
        block_masks.append(mask)
    
    # Initial active set: all blocks
    active = list(range(len(all_blocks)))
    
    # Solution storage
    solution = []
    
    def solve(active_indices, covered_mask, current_blocks):
        """
        Recursive backtracking.
        active_indices: list of indices of blocks that do not conflict with current_blocks
        covered_mask: integer bitmask of covered 5-subsets
        current_blocks: list of block indices selected so far
        """
        if time.time() > deadline:
            return False
        
        if covered_mask == ALL_ONES:
            solution.extend(current_blocks)
            return True
        
        # Find the first uncovered column
        # We can optimize by finding the column with min degree in active_indices,
        # but for this small problem, first uncovered is often sufficient.
        # To be safe and fast, let's find the uncovered column with the fewest active rows.
        
        # This step is O(num_cols * len(active_indices)) in worst case if done naively.
        # 792 * 924 ~ 700k ops.
        # We do this at each of 132 levels.
        # 132 * 700k = 90M ops. Might be too slow in Python.
        
        # Optimization: Just pick the first uncovered column.
        # The problem is highly symmetric.
        
        # Find first bit not in covered_mask
        # We can use bit manipulation to find the lowest set bit in ~covered_mask
        # ~covered_mask is infinite in Python, so we mask it.
        remaining = ALL_ONES ^ covered_mask
        if remaining == 0:
             # Should have been caught above
             return False
             
        # Find index of lowest set bit
        # (remaining & -remaining).bit_length() - 1
        lowest_bit = remaining & -remaining
        col_idx = lowest_bit.bit_length() - 1
        
        # Find all active blocks covering this column
        candidates = []
        for i in active_indices:
            if (block_masks[i] >> col_idx) & 1:
                candidates.append(i)
                
        # Heuristic: Try candidates in order of how many other columns they cover?
        # Or just iterate.
        
        for i in candidates:
            m = block_masks[i]
            # New covered mask
            new_covered = covered_mask | m
            
            # Filter active_indices: keep only those that do not intersect with m
            # Intersection means (mask[j] & m) != 0
            new_active = []
            for j in active_indices:
                if (block_masks[j] & m) == 0:
                    new_active.append(j)
            
            # Recurse
            if solve(new_active, new_covered, current_blocks + [i]):
                return True
                
        return False

    # Start solving
    # We need to handle the case where the first choice leads to a dead end.
    # The solver above does backtracking.
    
    if solve(active, 0, []):
        # Convert indices to actual blocks
        return [list(all_blocks[i]) for i in solution]
    else:
        # Fallback: Return whatever we have or empty
        return [list(all_blocks[i]) for i in solution] if solution else []

if __name__ == "__main__":
    blocks = construct()
    print(f"Found {len(blocks)} blocks")
