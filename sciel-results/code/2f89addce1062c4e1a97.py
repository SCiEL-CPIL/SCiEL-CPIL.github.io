def construct():
    n = 10
    
    # Try to find 2 MOLS
    # 1. Create a candidate L1
    # Use a standard Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # 2. Try to find L2 orthogonal to L1
    L2 = [[None for _ in range(n)] for _ in range(n)]
    used_pairs = [set() for _ in range(n)] # used_pairs[j] stores symbols used in column j? 
    # Better: global set of used (s1, s2) pairs.
    
    def solve(i):
        if i == n:
            return True
        # Try all permutations for row i? No, too slow.
        # Try all valid values for L2[i][j] one by one.
        # Heuristic: Fill row i.
        # We can use a list of available symbols for each column.
        
        # Simpler: Iterate through all permutations of 0..n-1 for row i?
        # 10! = 3.6M. Too slow for each row?
        # But we have constraints.
        
        # Let's use a simple backtracking on cells.
        # Order: Fill row by row.
        # For row i, we need to assign values such that:
        # - Each value 0..9 used once in row.
        # - For each col j, value not used in col j so far.
        # - For each col j, pair (L1[i][j], val) not used so far.
        
        # To make it faster, we can pre-compute available symbols for each col j at step i.
        
        def backtrack(row, col, row_vals, col_used, pair_used):
            if col == n:
                if row == n - 1:
                    return True
                return backtrack(row + 1, 0, set(), col_used, pair_used)
            
            # Get available symbols for this cell (row, col)
            avail = set(range(n))
            avail.difference_update(row_vals)
            avail.difference_update(col_used[col])
            
            # Check pair constraint
            s1 = L1[row][col]
            valid_vals = []
            for v in avail:
                if (s1, v) not in pair_used:
                    valid_vals.append(v)
                    
            # Shuffle for randomness to escape dead ends
            random.shuffle(valid_vals)
            
            for v in valid_vals:
                # Assign
                L2[row][col] = v
                row_vals.add(v)
                col_used[col].add(v)
                pair_used.add((s1, v))
                
                if backtrack(row, col + 1, row_vals, col_used, pair_used):
                    return True
                
                # Unassign
                row_vals.remove(v)
                col_used[col].remove(v)
                pair_used.remove((s1, v))
                
            return False

        # Init
        col_used = [set() for _ in range(n)]
        pair_used = set()
        
        if backtrack(0, 0, set(), col_used, pair_used):
            return [L1, L2]
        else:
            # If fails, try a different L1?
            # For now, return L1 only if no pair found.
            return [L1]

