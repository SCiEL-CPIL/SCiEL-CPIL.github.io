import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) with 132 blocks.
    Uses bitmask backtracking with symmetry breaking.
    """
    START = time.time()
    TIME_LIMIT = 140.0
    
    points = list(range(12))
    all_6_subsets = list(combinations(points, 6))
    all_5_subsets = list(combinations(points, 5))
    
    s5_index = {s5: i for i, s5 in enumerate(all_5_subsets)}
    num_cols = len(all_5_subsets) # 792
    num_rows = len(all_6_subsets) # 924
    
    # Identify the row corresponding to (0, 1, 2, 3, 4, 5) for symmetry breaking
    target_block = (0, 1, 2, 3, 4, 5)
    target_row_idx = all_6_subsets.index(target_block)
    
    # Precompute row masks
    row_masks = [0] * num_rows
    for i, b in enumerate(all_6_subsets):
        mask = 0
        for s5 in combinations(b, 5):
            mask |= (1 << s5_index[s5])
        row_masks[i] = mask
        
    # Precompute col_to_rows
    col_to_rows = [[] for _ in range(num_cols)]
    for i, mask in enumerate(row_masks):
        m = mask
        while m:
            c = (m & -m).bit_length() - 1
            col_to_rows[c].append(i)
            m ^= (m & -m)
            
    solution = []
    used_rows = [False] * num_rows
    
    def backtrack(covered_mask, depth):
        if time.time() - START > TIME_LIMIT:
            return False
            
        if covered_mask == (1 << num_cols) - 1:
            return True
            
        # Symmetry Breaking: At depth 0, force the target block
        if depth == 0:
            r_id = target_row_idx
            if used_rows[r_id]:
                return False
            used_rows[r_id] = True
            solution.append(r_id)
            if backtrack(covered_mask | row_masks[r_id], depth + 1):
                return True
            solution.pop()
            used_rows[r_id] = False
            return False

        # Find the uncovered column with the fewest available rows (MRV)
        min_count = float('inf')
        min_col = -1
        candidates = []
        
        for c in range(num_cols):
            if not (covered_mask & (1 << c)):
                count = 0
                cand_list = []
                for r_id in col_to_rows[c]:
                    if used_rows[r_id]:
                        continue
                    if row_masks[r_id] & covered_mask:
                        continue
                    count += 1
                    cand_list.append(r_id)
                    
                if count < min_count:
                    min_count = count
                    min_col = c
                    candidates = cand_list
                    if count == 0:
                        return False
                    if count == 1:
                        break
                    
        if min_col == -1:
            return True
            
        for r_id in candidates:
            r_mask = row_masks[r_id]
            used_rows[r_id] = True
            solution.append(r_id)
            
            if backtrack(covered_mask | r_mask, depth + 1):
                return True
                
            solution.pop()
            used_rows[r_id] = False
            
        return False

    if backtrack(0, 0):
        return [list(all_6_subsets[r]) for r in solution]
    else:
        return []
