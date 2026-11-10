import time
import sys

DEADLINE = 85.0

def construct():
    start_time = time.time()
    n = 10
    
    # We will construct 9 MOLS of order 10.
    
    # 1. First square (cyclic)
    s1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # 2. Initialize other squares
    # We want 9 squares total.
    num_squares = 9
    squares = [s1]
    for _ in range(num_squares - 1):
        squares.append([[-1 for _ in range(n)] for _ in range(n)])
    
    # 3. Backtracking solver
    
    # Optimization: Fix the first row of the second square (index 1) to 0, 1, ..., 9.
    # This is possible via symbol permutation.
    for j in range(n):
        squares[1][0][j] = j
    
    # We start filling square 1 (index 1) at cell (1, 0).
    # Note: Row 0 is already filled.
    
    def is_valid(idx, i, j, val):
        sq = squares[idx]
        
        # 1. Latin Square Constraints
        # Row i
        for k in range(n):
            if k != j and sq[i][k] == val:
                return False
        # Col j
        for k in range(n):
            if k != i and sq[k][j] == val:
                return False
        
        # 2. Orthogonality with previous squares
        for prev_idx in range(idx):
            prev_sq = squares[prev_idx]
            # The pair (prev_sq[i][j], val) must be unique in the grid.
            # Check all other cells (r, c)
            for r in range(n):
                for c in range(n):
                    if r == i and c == j:
                        continue
                    if sq[r][c] == -1:
                        continue
                    if prev_sq[r][c] == prev_sq[i][j] and sq[r][c] == val:
                        return False
        
        return True
    
    def backtrack(idx, i, j):
        if time.time() - start_time > DEADLINE:
            return False
            
        if idx >= num_squares:
            return True
            
        if i == n:
            if j != 0:
                return False
            # Move to next square
            # Start at (0, 0) of next square?
            # But we might have normalized the first row of the next square?
            # We can only normalize the first square we build (index 1).
            # For subsequent squares, we cannot fix the first row without loss of generality?
            # Actually, we can apply symbol permutations to ANY new square we are building.
            # So for square idx > 1, we can also fix its first row to 0..9?
            # Yes, because we can permute the symbols of the new square independently.
            # So we can fix the first row of EVERY square to 0..9.
            
            # If we are starting a new square (idx > 1), we should fix its first row.
            if idx > 1:
                # Fix first row of squares[idx]
                for jj in range(n):
                    squares[idx][0][jj] = jj
                # Start at (1, 0)
                return backtrack(idx, 1, 0)
            else:
                # This case shouldn't happen if we handle idx=1 start correctly
                return backtrack(idx, 0, 0)
        
        if j == n:
            return backtrack(idx, i + 1, 0)
        
        # If cell is already filled
        if squares[idx][i][j] != -1:
            return backtrack(idx, i, j + 1)
        
        # Try all values
        for val in range(n):
            if is_valid(idx, i, j, val):
                squares[idx][i][j] = val
                if backtrack(idx, i, j + 1):
                    return True
                squares[idx][i][j] = -1
        
        return False
    
    # Start backtracking for square 1 at (1, 0) because row 0 is fixed.
    if backtrack(1, 1, 0):
        # Verify result
        # (Optional but good for safety)
        pass
    else:
        # Fallback
        return [s1]
    
    # Convert to proper types
    result = []
    for sq in squares:
        new_sq = []
        for row in sq:
            new_row = [int(v) for v in row]
            new_sq.append(new_row)
        result.append(new_sq)
        
    return result
