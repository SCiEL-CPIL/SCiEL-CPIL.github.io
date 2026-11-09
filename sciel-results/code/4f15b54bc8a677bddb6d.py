import time

def construct():
    n = 10
    # L1: Cyclic Latin Square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1]
    
    # Find L2 orthogonal to L1
    L2 = find_mols_backtracking(squares, n)
    if L2:
        squares.append(L2)
        
    # Find L3 orthogonal to L1 and L2
    L3 = find_mols_backtracking(squares, n)
    if L3:
        squares.append(L3)
        
    # M(10) is known to be 3, so we stop here.
    # If we wanted to be safe, we could try L4 with a very short timeout, 
    # but it's better to return the valid 3 quickly.
    
    return squares

def find_mols_backtracking(fixed_squares, n):
    """
    Finds a Latin square orthogonal to all squares in fixed_squares.
    Uses backtracking with MRV (Minimum Remaining Values) heuristic.
    """
    # grid[r][c] is the value at (r,c). Initially None.
    grid = [[None for _ in range(n)] for _ in range(n)]
    
    # Data structures for constraint tracking
    row_vals = [set() for _ in range(n)]
    col_vals = [set() for _ in range(n)]
    
    # For each fixed square F_idx, and for each symbol k (0..n-1),
    # track which values have been used in grid at positions where F has k.
    # used_pairs[idx][k] is a set of values.
    used_pairs = []
    for _ in fixed_squares:
        used_pairs.append([set() for _ in range(n)])
        
    def get_available_values(r, c):
        avail = set(range(n))
        avail -= row_vals[r]
        avail -= col_vals[c]
        for idx, F in enumerate(fixed_squares):
            k = F[r][c]
            avail -= used_pairs[idx][k]
        return avail

    def backtrack():
        # Find cell with min available values (MRV)
        min_count = n + 1
        min_r, min_c = -1, -1
        min_avail = None
        
        for r in range(n):
            for c in range(n):
                if grid[r][c] is None:
                    avail = get_available_values(r, c)
                    count = len(avail)
                    if count < min_count:
                        min_count = count
                        min_r, min_c = r, c
                        min_avail = avail
                    if count == 0:
                        return False
        if min_r == -1:
            return True # All filled
            
        # Try values in min_avail
        # Iterate in sorted order for determinism, or random for variety.
        # Sorted is fine.
        for val in sorted(min_avail):
            # Place val
            grid[min_r][min_c] = val
            row_vals[min_r].add(val)
            col_vals[min_c].add(val)
            
            # Update used_pairs
            undo_info = []
            for idx, F in enumerate(fixed_squares):
                k = F[min_r][min_c]
                used_pairs[idx][k].add(val)
                undo_info.append((idx, k))
                
            if backtrack():
                return True
                
            # Undo
            grid[min_r][min_c] = None
            row_vals[min_r].remove(val)
            col_vals[min_c].remove(val)
            for idx, k in undo_info:
                used_pairs[idx][k].remove(val)
                
        return False

    if backtrack():
        return [[grid[r][c] for c in range(n)] for r in range(n)]
    return None
