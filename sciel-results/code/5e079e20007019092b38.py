import time
import sys
from itertools import combinations

def construct():
    """
    Construct the Witt design S(5,6,12) using Algorithm X (DLX) for Exact Cover
    with symmetry breaking to fix the first few blocks.
    """
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    col_index = {s: i for i, s in enumerate(all_5subsets)}
    num_cols = len(all_5subsets)  # 792
    
    all_6subsets = list(combinations(points, 6))
    row_index = {b: i for i, b in enumerate(all_6subsets)}
    rows = []
    for b in all_6subsets:
        cols = [col_index[s] for s in combinations(b, 5)]
        rows.append(cols)
        
    # Precompute for each column, which rows cover it
    col_to_rows = [[] for _ in range(num_cols)]
    for r_idx, cols in enumerate(rows):
        for c in cols:
            col_to_rows[c].append(r_idx)
            
    # Symmetry breaking: Fix the first block to {0,1,2,3,4,5}
    # This reduces the search space by fixing one block
    fixed_block = (0, 1, 2, 3, 4, 5)
    fixed_row = row_index[fixed_block]
    
    # Mark columns covered by the fixed block
    covered = set()
    for c in rows[fixed_row]:
        covered.add(c)
        
    used_rows = {fixed_row}
    solution = [fixed_row]
    
    def search(depth):
        if len(covered) == num_cols:
            return True
            
        # Check time limit
        if time.time() > deadline:
            return False
            
        # Heuristic: Find the column with the minimum number of candidate rows
        min_count = float('inf')
        best_col = -1
        best_candidates = []
        
        for c in range(num_cols):
            if c in covered:
                continue
                
            candidates = []
            for r in col_to_rows[c]:
                if r in used_rows:
                    continue
                # Check if row r conflicts with current covered set
                conflict = False
                for cc in rows[r]:
                    if cc in covered:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(r)
            
            if len(candidates) < min_count:
                min_count = len(candidates)
                best_col = c
                best_candidates = candidates
                
                if min_count == 0:
                    return False
                if min_count == 1:
                    break  # Can't get better than 1
            
        if best_col == -1:
            return False  # No uncovered columns? Should be handled by base case.
            
        # Try each candidate row for best_col
        for r in best_candidates:
            # Add row r
            used_rows.add(r)
            new_covered_cols = [cc for cc in rows[r] if cc not in covered]
            for cc in new_covered_cols:
                covered.add(cc)
            
            solution.append(r)
            
            if search(depth + 1):
                return True
            
            # Backtrack
            solution.pop()
            used_rows.remove(r)
            for cc in new_covered_cols:
                covered.remove(cc)
                
        return False

    # Increase recursion limit just in case
    sys.setrecursionlimit(10000)
    
    if search(0):
        # Convert row indices to blocks
        result = []
        for r_idx in solution:
            result.append(list(all_6subsets[r_idx]))
        return result
    else:
        # Fallback: return empty or a partial solution if search fails
        return []

# EVOLVE-BLOCK-START
# The function construct is defined above.
# EVOLVE-BLOCK-END
