import time
import sys
sys.setrecursionlimit(10000)

def construct():
    n = 10
    # L1: Cyclic Latin Square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [L1]
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s for overhead
    
    # Try to add squares one by one
    # We use a backtracking CSP solver for each new square
    # The domain of each cell is 0..9
    # Constraints:
    # 1. Row uniqueness
    # 2. Col uniqueness
    # 3. Orthogonality with all existing squares
    
    for k in range(2, 10):
        if time.time() - start_time > time_limit:
            break
            
        new_square = solve_latin_square_csp(squares, n, timeout=time_limit - (time.time() - start_time))
        if new_square:
            squares.append(new_square)
        else:
            # If we fail to find k-th square, we can stop.
            # It is unlikely we can find a k+1-th square if we can't find k-th?
            # Actually, if we fail to find L2, we return 1.
            # If we find L2 but fail L3, we return 2.
            break
            
    return squares

def solve_latin_square_csp(fixed_squares, n, timeout):
    # Variables: (row, col)
    # Domain: 0..n-1
    # We fill the grid cell by cell.
    # Order: We can use a static order or dynamic. Static is simpler.
    # To improve, we can choose the cell with the smallest domain (MRV).
    
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Pre-compute used symbols in rows and cols for speed?
    # We will maintain row_used and col_used as sets or counts.
    row_counts = [[0]*n for _ in range(n)]
    col_counts = [[0]*n for _ in range(n)]
    
    # Orthogonality constraints:
    # For each fixed square L_fix, the pair (L_fix[r][c], L_new[r][c]) must be unique.
    # We can maintain a set of used pairs for each fixed square.
    pair_counts = [ [ [0]*n for _ in range(n) ] for _ in range(len(fixed_squares)) ]
    # pair_counts[k][a][b] = number of times pair (a,b) has been used for fixed square k
    
    # Actually, it's better to check validity on the fly.
    
    def is_valid(r, c, val):
        # Check row
        if row_counts[r][val] > 0:
            return False
        # Check col
        if col_counts[c][val] > 0:
            return False
        # Check orthogonality
        for idx, L_fix in enumerate(fixed_squares):
            fix_val = L_fix[r][c]
            if pair_counts[idx][fix_val][val] > 0:
                return False
        return True

    def assign(r, c, val):
        grid[r][c] = val
        row_counts[r][val] += 1
        col_counts[c][val] += 1
        for idx, L_fix in enumerate(fixed_squares):
            fix_val = L_fix[r][c]
            pair_counts[idx][fix_val][val] += 1

    def unassign(r, c, val):
        grid[r][c] = -1
        row_counts[r][val] -= 1
        col_counts[c][val] -= 1
        for idx, L_fix in enumerate(fixed_squares):
            fix_val = L_fix[r][c]
            pair_counts[idx][fix_val][val] -= 1

    def get_domain(r, c):
        # Return list of valid values
        dom = []
        for val in range(n):
            if is_valid(r, c, val):
                dom.append(val)
        return dom

    # Backtracking search
    # We need to track time to abort.
    start = time.time()
    
    # We will use a simple recursive backtracker.
    # To avoid deep recursion stack issues, we can use iterative or just set limit.
    # For n=10, depth is 100. Recursion is fine.
    
    def backtrack(r, c):
        if time.time() - start > timeout:
            return False
            
        if r == n - 1 and c == n - 1:
            return True
            
        # Next cell
        nr, nc = r, c + 1
        if nc == n:
            nr, nc = r + 1, 0
            
        # Optimization: MRV heuristic is complex to implement dynamically in a simple backtracker
        # without rebuilding the domain list for all cells.
        # For now, static order (row-major) is often sufficient for Latin Squares 
        # if the first row is fixed or constrained.
        
        # Let's try to pick a value from the domain
        dom = get_domain(r, c)
        
        # Heuristic: Try values that appear less often in the current row/col? 
        # Or just random shuffle to avoid bias?
        # Randomization can help escape bad paths.
        import random
        random.shuffle(dom)
        
        for val in dom:
            assign(r, c, val)
            if backtrack(nr, nc):
                return True
            unassign(r, c, val)
            
        return False

    # To improve performance, we can fix the first row to be 0..9-1?
    # No, that reduces symmetry but might miss solutions if the first row isn't sorted.
    # However, any Latin square can be permuted to have the first row sorted.
    # Since we only care about existence, we can assume WLOG that the first row is 0..9.
    # This reduces the search space by 10!
    
    # Assign first row
    for c in range(n):
        val = c
        if not is_valid(0, c, val):
            # This shouldn't happen if we start with empty counts
            # But wait, orthogonality might block this.
            # If we can't place c at (0,c), then this WLOG assumption is wrong for this specific instance?
            # No, we can permute the symbols of the NEW square.
            # So we can always relabel the symbols of L_new such that L_new[0][j] = j.
            # This is a valid transformation.
            # So we CAN force the first row to be 0..n-1.
            pass
            
    # Force first row
    for c in range(n):
        val = c
        # Check if valid
        if not is_valid(0, c, val):
             # If not valid, it means the orthogonality constraint is violated?
             # (L_fix[0][c], c) is already used?
             # If so, we need to permute the symbols differently.
             # This simple WLOG might fail if the fixed squares don't allow identity mapping on row 0.
             # To be safe, we should not force it, or we should try all permutations of row 0?
             # 10! is too big.
             
             # Let's just use the generic backtracker without forcing row 0.
             # But we can shuffle the domain.
             return None # If we can't even start with the simple check? No, we just don't force it.

    # Let's restart the logic without forcing row 0, but using randomization in backtracking.
    
    # Actually, forcing the first row to be 0..9 is a very strong constraint.
    # If it fails, it means no solution exists with that first row.
    # But a solution exists with *some* first row.
    # Since we can permute symbols, there is a solution where the first row is 0..9?
    # Yes, take any solution L, and apply a permutation P to its symbols such that P(L[0][j]) = j.
    # This is always possible because L[0] is a permutation of 0..9.
    # So we CAN force the first row to be 0..9.
    
    # Let's try to assign the first row.
    # If any assignment in the first row fails `is_valid`, then the WLOG assumption is incorrect?
    # No, `is_valid` checks if the pair (L_fix[0][c], c) is unique.
    # If for some c, the pair (L_fix[0][c], c) is already used in the first row?
    # No, we are assigning the first row cell by cell.
    # When we assign (0,0) to 0, we check if (L_fix[0][0], 0) is used. It's not.
    # When we assign (0,1) to 1, we check if (L_fix[0][1], 1) is used.
    # The only conflict is if L_fix[0][c] == L_fix[0][c'] and c == c'? No.
    # The pair is (fix_val, new_val).
    # If L_fix[0] is a Latin row, fix_val is unique for each col.
    # So (L_fix[0][c], c) are all distinct pairs because the second element is distinct.
    # So the first row assignment 0..9 is ALWAYS valid with respect to orthogonality?
    # Yes, because the second element of the pair is distinct for each column.
    # So we can safely force the first row to be 0..9.
    
    for c in range(n):
        val = c
        assign(0, c, val)
        
    # Now backtrack from (1, 0)
    if backtrack(1, 0):
        # Success
        # Convert grid to list of lists of ints
        return [ [int(grid[r][c]) for c in range(n)] for r in range(n) ]
    else:
        return None
