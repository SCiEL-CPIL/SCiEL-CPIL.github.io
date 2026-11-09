import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) using Dancing Links (DLX) 
    for Exact Cover. Returns a list of 132 distinct 6-subsets of 0..11.
    """
    START = time.time()
    TIME_LIMIT = 140.0
    
    points = list(range(12))
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    # Map 5-subsets to indices
    s5_index = {s5: i for i, s5 in enumerate(all_5_subsets)}
    num_cols = len(all_5_subsets)
    
    # Build the matrix: rows are 6-subsets, cols are 5-subsets they cover
    # Each row has 5 columns
    rows = []
    for b in all_6_subsets:
        cols = [s5_index[s5] for s5 in combinations(b, 5)]
        rows.append(cols)
        
    # DLX Implementation
    # We use a simple array-based approach for speed in Python
    # Size: max rows * max cols per row + 1 (header)
    # Actually, standard DLX uses linked lists. 
    # For small problems, a recursive search with bitmasks might be faster than implementing full DLX structures in Python.
    
    # Alternative: Bitmask backtracking
    # State: covered columns (bitmask of 792 bits -> use Python int)
    # Rows: list of (cols_mask, row_id)
    
    # Precompute masks for each row
    row_masks = []
    for r in rows:
        mask = 0
        for c in r:
            mask |= (1 << c)
        row_masks.append(mask)
        
    # Sort rows by number of 1s (all 5) or by frequency of columns?
    # Heuristic: Choose column with fewest options.
    
    # Precompute for each column, which rows contain it
    col_to_rows = [[] for _ in range(num_cols)]
    for i, r in enumerate(rows):
        for c in r:
            col_to_rows[c].append(i)
            
    # We need to track which rows are used.
    # Since we need 132 rows out of 924, and they must be disjoint in columns.
    
    def solve(covered_mask, used_rows_mask, solution):
        if time.time() - START > TIME_LIMIT:
            return False
            
        if covered_mask == (1 << num_cols) - 1:
            return True
            
        # Find column with minimum available rows
        # To speed up, we can pick the first uncovered column with min candidates
        # But checking all 792 columns every step is slow.
        
        # Heuristic: Pick a random uncovered column? No, deterministic.
        # Pick the column with the smallest number of *available* rows.
        
        # Optimization: If we just pick the first uncovered column, it might be slow.
        # Let's use the "most constrained" heuristic properly.
        
        min_count = float('inf')
        min_col = -1
        min_candidates = None
        
        # Iterate through uncovered columns
        # We can maintain a list of uncovered columns
        # For simplicity in this edit, we'll scan. 792 is small enough?
        # 132 steps * 792 cols * (check candidates) -> ~100k ops per step? No.
        
        # Better: Use a priority queue or just scan.
        # Given the size, scanning 792 columns at each of 132 steps is ~100k iterations.
        # Inside, we check candidates.
        
        for c in range(num_cols):
            if not (covered_mask & (1 << c)):
                # Count available rows for this column
                # A row is available if it is not used and its other cols are not covered?
                # No, in Exact Cover, if we pick a row, all its cols become covered.
                # So a row is a candidate for col c if:
                # 1. Row contains c
                # 2. Row is not used
                # 3. No other column in the row is already covered? 
                #    Actually, if another col in the row is covered, the row is invalid because it would cover an already covered col twice.
                #    So yes, the row must be "disjoint" from the current covered set except for c?
                #    No, if we pick row R for col c, R must not cover any *other* covered col.
                
                count = 0
                candidates = []
                for r_id in col_to_rows[c]:
                    if used_rows_mask & (1 << r_id):
                        continue
                    r_mask = row_masks[r_id]
                    # Check if r_mask intersects with covered_mask excluding c?
                    # If r_mask & (covered_mask ^ (1 << c)) != 0, then it conflicts.
                    # Actually, if r_mask & covered_mask has more than 1 bit (c), it conflicts.
                    # Since c is uncovered, r_mask & covered_mask should be 0 for a valid candidate?
                    # No, c is in r_mask. c is NOT in covered_mask.
                    # So r_mask & covered_mask should be 0.
                    if r_mask & covered_mask:
                        continue
                    count += 1
                    candidates.append(r_id)
                    
                if count < min_count:
                    min_count = count
                    min_col = c
                    min_candidates = candidates
                    if count == 0:
                        return False
                    if count == 1:
                        break
                        
        if min_col == -1:
            return True # Should not happen if not all covered
            
        for r_id in min_candidates:
            r_mask = row_masks[r_id]
            new_covered = covered_mask | r_mask
            new_used = used_rows_mask | (1 << r_id)
            solution.append(r_id)
            if solve(new_covered, new_used, solution):
                return True
            solution.pop()
            
        return False

    solution = []
    # Initial mask 0
    # Note: used_rows_mask is an int with 924 bits. Python handles this.
    
    if solve(0, 0, solution):
        return [list(all_6_subsets[r]) for r in solution]
    else:
        # Fallback
        return []
