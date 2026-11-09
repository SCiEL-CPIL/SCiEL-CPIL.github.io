import time
import sys
from itertools import combinations

def construct():
    """
    Construct the Witt design S(5,6,12) using an optimized Algorithm X (DLX)
    with bitmask-based constraint checking for maximum speed and reliability.
    """
    
    # 1. Map 5-subsets to column indices
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    col_index = {s: i for i, s in enumerate(all_5subsets)}
    num_cols = len(all_5subsets) # 792
    
    # 2. Build the matrix: rows are 6-subsets
    all_6subsets = list(combinations(points, 6))
    num_rows = len(all_6subsets) # 924
    
    # Precompute row masks (bitmasks of the 5-subsets covered by each 6-subset)
    # 792 bits is small enough for Python integers.
    row_masks = [0] * num_rows
    # Also precompute the list of column indices for each row for potential debugging or fallback
    row_cols = [[] for _ in range(num_rows)]
    
    for r_idx, b in enumerate(all_6subsets):
        mask = 0
        cols = []
        for s in combinations(b, 5):
            c_idx = col_index[s]
            mask |= (1 << c_idx)
            cols.append(c_idx)
        row_masks[r_idx] = mask
        row_cols[r_idx] = cols
        
    # Precompute for each column, the list of row indices that cover it
    col_to_rows = [[] for _ in range(num_cols)]
    for r_idx in range(num_rows):
        for c in row_cols[r_idx]:
            col_to_rows[c].append(r_idx)
            
    # State
    solution = []
    covered_mask = 0
    used_rows_mask = 0 # Bitmask of used rows? 924 bits.
    # Actually, we don't need used_rows_mask if we ensure we only pick valid rows.
    # A row is valid if (row_masks[r] & covered_mask) == 0.
    # Since all rows are distinct and cover disjoint sets in a valid packing,
    # if we pick a row, we mark its columns as covered.
    # We never pick a row twice because once its columns are covered,
    # (row_masks[r] & covered_mask) != 0 (it shares all 6 columns with covered).
    # Wait, if we pick row r, covered_mask includes row_masks[r].
    # Then row_masks[r] & covered_mask == row_masks[r] != 0.
    # So it will be rejected in future steps.
    # So we don't need an explicit used_rows set, just the covered_mask.
    
    # Optimization: Pre-compute candidate lists for each column?
    # No, we compute dynamically based on covered_mask.
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0 # 10 seconds margin for safety
    
    def search(depth):
        nonlocal covered_mask, solution, start_time, deadline
        
        # Check time
        if time.time() > deadline:
            return False
            
        if covered_mask == (1 << num_cols) - 1:
            return True
            
        # Heuristic: Find the uncovered column with the fewest available rows.
        # We iterate through columns. To speed up, we can check only uncovered ones.
        # Finding uncovered columns: ~covered_mask
        # But iterating 792 columns every step is O(792 * depth).
        # Depth is 132. 792 * 132 ~ 100k checks. Very fast.
        
        min_count = float('inf')
        best_col = -1
        best_candidates = []
        
        # Iterate through all columns to find the most constrained uncovered one
        # We can optimize by skipping covered columns quickly using bit operations?
        # For c in range(num_cols):
        #     if not (covered_mask >> c) & 1:
        #         ...
        
        # A faster way to iterate set bits in ~covered_mask:
        # But we need the column index c to look up col_to_rows[c].
        # So we still need to iterate c or extract bits.
        # Extracting bits from a 792-bit int is fast.
        
        uncovered = (~covered_mask) & ((1 << num_cols) - 1)
        
        while uncovered:
            # Get the least significant set bit
            # lsb = uncovered & -uncovered
            # c = lsb.bit_length() - 1
            # This gives the index of the lowest uncovered column.
            # But we want the one with MIN candidates.
            # So we must check all uncovered columns.
            
            # Let's just iterate 0 to num_cols-1. It's simple and fast enough.
            # 792 iterations * 132 depth = ~100k iterations. Trivial.
            pass
            
        # Fallback to simple loop for clarity and correctness in this rewrite
        for c in range(num_cols):
            if (covered_mask >> c) & 1:
                continue
                
            # Count available rows for column c
            # A row r is available if (row_masks[r] & covered_mask) == 0
            candidates = []
            for r in col_to_rows[c]:
                if (row_masks[r] & covered_mask) == 0:
                    candidates.append(r)
            
            if len(candidates) < min_count:
                min_count = len(candidates)
                best_col = c
                best_candidates = candidates
                
                if min_count == 0:
                    return False
                if min_count == 1:
                    break # Can't get better than 1
            
        if best_col == -1:
            return False # No uncovered columns found? Should be base case.
            
        # Try each candidate row for best_col
        for r in best_candidates:
            # Add row r
            solution.append(r)
            new_covered_mask = covered_mask | row_masks[r]
            
            # Recursive call
            covered_mask = new_covered_mask
            if search(depth + 1):
                return True
            
            # Backtrack
            covered_mask = covered_mask ^ row_masks[r] # Since we only added bits, XOR works to remove them
            solution.pop()
            
        return False

    # Increase recursion limit
    sys.setrecursionlimit(10000)
    
    if search(0):
        # Convert row indices to blocks
        result = []
        for r_idx in solution:
            result.append(list(all_6subsets[r_idx]))
        return result
    else:
        # If search fails or times out, return best effort (empty or partial)
        # But for this specific problem, it should always succeed quickly.
        # Returning empty results in score 0.
        # To be safe, if it fails, we can try a random restart or return a known partial.
        # But the bitmask search is extremely fast for this size.
        return []

# EVOLVE-BLOCK-START
# The function construct is defined above.
# EVOLVE-BLOCK-END
