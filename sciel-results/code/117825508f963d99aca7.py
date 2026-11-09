import random
import time

def construct():
    n = 10
    # L1: Cyclic Latin Square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [L1]
    
    start_time = time.time()
    max_time = 55.0 # Leave 5s for overhead
    
    # We try to find up to 8 more squares (total 9)
    # Order 10 has 9 MOLS.
    
    for _ in range(8):
        if time.time() - start_time > max_time:
            break
            
        # Try to find a square orthogonal to all existing squares
        # Using a backtracking solver with MRV
        new_square = solve_mols(squares, n, timeout=5.0)
        
        if new_square is not None:
            squares.append(new_square)
        else:
            # If we fail to find one within timeout, we stop.
            break
            
    return squares

def solve_mols(fixed_squares, n, timeout=5.0):
    """
    Finds a Latin Square L such that L is orthogonal to all squares in fixed_squares.
    Uses backtracking with MRV (Minimum Remaining Values).
    """
    start = time.time()
    
    # Pre-compute constraints
    # For each cell (r, c), the value v is invalid if:
    # 1. v is already in row r
    # 2. v is already in col c
    # 3. For any fixed square S, the pair (S[r][c], v) has already been used in another cell.
    
    # We will maintain:
    # grid: n x n, None if unassigned
    # row_used: n x n boolean matrix (row_used[r][v] = True if v in row r)
    # col_used: n x n boolean matrix (col_used[c][v] = True if v in col c)
    # pair_used: list of sets, one for each fixed square.
    # pair_used[k] tracks which pairs (val_fixed, val_new) have been used.
    
    grid = [[None for _ in range(n)] for _ in range(n)]
    row_used = [[False] * n for _ in range(n)]
    col_used = [[False] * n for _ in range(n)]
    used_pairs = [set() for _ in range(len(fixed_squares))]
    
    def is_valid(r, c, v):
        if row_used[r][v]:
            return False
        if col_used[c][v]:
            return False
        for k, S in enumerate(fixed_squares):
            pair = (S[r][c], v)
            if pair in used_pairs[k]:
                return False
        return True

    def assign(r, c, v):
        grid[r][c] = v
        row_used[r][v] = True
        col_used[c][v] = True
        for k, S in enumerate(fixed_squares):
            used_pairs[k].add((S[r][c], v))

    def unassign(r, c, v):
        grid[r][c] = None
        row_used[r][v] = False
        col_used[c][v] = False
        for k, S in enumerate(fixed_squares):
            used_pairs[k].remove((S[r][c], v))

    def count_options(r, c):
        count = 0
        for v in range(n):
            if is_valid(r, c, v):
                count += 1
        return count

    def backtrack():
        if time.time() - start > timeout:
            return False

        # Find cell with MRV (Minimum Remaining Values)
        min_count = n + 1
        min_r, min_c = -1, -1
        
        for r in range(n):
            for c in range(n):
                if grid[r][c] is None:
                    cnt = count_options(r, c)
                    if cnt == 0:
                        return False # Dead end
                    if cnt < min_count:
                        min_count = cnt
                        min_r, min_c = r, c
                        if min_count == 1:
                            break
            if min_count == 1:
                break
                
        if min_r == -1:
            return True # All assigned
        
        r, c = min_r, min_c
        # Shuffle values to add randomness and avoid getting stuck in bad branches
        vals = list(range(n))
        random.shuffle(vals)
        
        for v in vals:
            if is_valid(r, c, v):
                assign(r, c, v)
                if backtrack():
                    return True
                unassign(r, c, v)
                
        return False

    if backtrack():
        return grid
    return None
