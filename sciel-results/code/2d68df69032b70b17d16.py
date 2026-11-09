import sys
import time
from itertools import combinations

def construct():
    """
    Construct the Witt design S(5,6,12) using the relationship with S(4,5,11).
    Optimized with bitmasks and a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    # 1. Solve for S(4,5,11)
    points_11 = range(11)
    all_4subsets = list(combinations(points_11, 4))
    col_index_4 = {s: i for i, s in enumerate(all_4subsets)}
    num_cols_4 = len(all_4subsets) # 330
    
    all_5subsets_11 = list(combinations(points_11, 5))
    rows_11 = []
    for b in all_5subsets_11:
        cols = [col_index_4[s] for s in combinations(b, 4)]
        rows_11.append(cols)
        
    col_to_rows_4 = [[] for _ in range(num_cols_4)]
    for r_idx, cols in enumerate(rows_11):
        for c in cols:
            col_to_rows_4[c].append(r_idx)
            
    solution_11 = []
    used_rows_11_mask = 0
    covered_4_mask = 0
    
    # Pre-compute row masks for faster conflict checking?
    # rows_11[r] is a list of col indices.
    # We can check conflict by seeing if (row_mask[r] & covered_4_mask) has any bit other than the target col.
    # But we need to check if *any* other col is covered.
    
    def search_11(depth):
        nonlocal used_rows_11_mask, covered_4_mask, solution_11
        
        if time.time() > deadline:
            return False
            
        if covered_4_mask == (1 << num_cols_4) - 1:
            return True
            
        min_count = float('inf')
        best_col = -1
        best_candidates = []
        
        # Find the column with the fewest candidates
        # Iterate through all cols to find MRV
        for c in range(num_cols_4):
            if (covered_4_mask >> c) & 1:
                continue
                
            candidates = []
            for r in col_to_rows_4[c]:
                if (used_rows_11_mask >> r) & 1:
                    continue
                # Check conflict: does row r cover any other covered col?
                # We need to check if (row_cols[r] & covered_4_mask) has any bit set that is NOT c.
                # Pre-compute row_masks?
                conflict = False
                for cc in rows_11[r]:
                    if cc != c and (covered_4_mask >> cc) & 1:
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
                    break
            
        if best_col == -1:
            return False
            
        for r in best_candidates:
            used_rows_11_mask |= (1 << r)
            new_covered_bits = 0
            for cc in rows_11[r]:
                if not ((covered_4_mask >> cc) & 1):
                    new_covered_bits |= (1 << cc)
            covered_4_mask |= new_covered_bits
            
            solution_11.append(r)
            
            if search_11(depth + 1):
                return True
            
            solution_11.pop()
            used_rows_11_mask &= ~(1 << r)
            covered_4_mask &= ~new_covered_bits
                
        return False

    sys.setrecursionlimit(10000)
    
    if not search_11(0):
        return []

    # 2. Construct S(5,6,12) from S(4,5,11)
    result = []
    V = set(range(11))
    for r_idx in solution_11:
        B = set(all_5subsets_11[r_idx])
        # Block 1: B U {11}
        result.append(sorted(list(B) + [11]))
        # Block 2: V \ B
        comp = V - B
        result.append(sorted(list(comp)))
        
    return result
