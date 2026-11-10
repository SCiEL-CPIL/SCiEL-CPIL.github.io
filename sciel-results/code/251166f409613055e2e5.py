"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n = 10
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Start with the cyclic Latin square
    s0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [s0]
    
    # Helper: check if a partial square can still be completed
    def is_valid_partial(square, row, col):
        """Check if placing value v at (row, col) is consistent with Latin square constraints."""
        v = square[row][col]
        if v < 0:
            return True
        # Check row
        for c in range(col):
            if square[row][c] == v:
                return False
        for c in range(col + 1, n):
            if square[row][c] == v:
                return False
        # Check col
        for r in range(row):
            if square[r][col] == v:
                return False
        for r in range(row + 1, n):
            if square[r][col] == v:
                return False
        return True
    
    def check_orthogonality_partial(square1, square2, row, col):
        """Check if the partial assignment at (row, col) doesn't violate orthogonality."""
        a = square1[row][col]
        b = square2[row][col]
        if a < 0 or b < 0:
            return True
        # Check if this pair already exists in the partial square
        for r in range(row):
            for c in range(n):
                if square1[r][c] == a and square2[r][c] == b:
                    return False
        for c in range(col):
            if square1[row][c] == a and square2[row][c] == b:
                return False
        return True
    
    def backtrack(square, existing_squares, row, col, time_limit):
        """
        Backtrack to fill the square. Returns True if completed successfully.
        square is modified in place.
        """
        if time.time() > time_limit:
            return False
        
        if row == n:
            return True
        
        if col == n:
            return backtrack(square, existing_squares, row + 1, 0, time_limit)
        
        # If cell is already filled, move on
        if square[row][col] >= 0:
            return backtrack(square, existing_squares, row, col + 1, time_limit)
        
        # Determine allowed values
        used_in_row = set(square[row])
        used_in_col = set(square[r][col] for r in range(n))
        allowed = [v for v in range(n) if v not in used_in_row and v not in used_in_col]
        
        # For orthogonality constraints, we need to be more careful
        # But for efficiency, let's just try values and check
        for v in allowed:
            square[row][col] = v
            
            # Check orthogonality with all existing squares
            valid = True
            for es in existing_squares:
                a = es[row][col]
                b = v
                # Check if pair (a, b) already exists in the partial square
                for r in range(row):
                    for c in range(n):
                        if es[r][c] == a and square[r][c] == b:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                # Check current row
                for c in range(col):
                    if es[row][c] == a and square[row][c] == b:
                        valid = False
                        break
                if not valid:
                    break
                # Check current column
                for r in range(row):
                    if es[r][col] == a and square[r][col] == b:
                        valid = False
                        break
            
            if valid:
                if backtrack(square, existing_squares, row, col + 1, time_limit):
                    return True
            
            square[row][col] = -1
        
        return False
    
    # Try to find additional squares
    max_squares = 9
    for target_idx in range(1, max_squares):
        if time.time() > deadline:
            break
        
        square = [[-1] * n for _ in range(n)]
        
        # Use a tighter time limit for each square
        square_deadline = min(deadline, time.time() + 40)
        
        success = backtrack(square, squares, 0, 0, square_deadline)
        
        if success:
            # Verify the complete square is valid
            valid = True
            for r in range(n):
                row_vals = [square[r][c] for c in range(n)]
                if sorted(row_vals) != list(range(n)):
                    valid = False
                    break
            if valid:
                for c in range(n):
                    col_vals = [square[r][c] for r in range(n)]
                    if sorted(col_vals) != list(range(n)):
                        valid = False
                        break
            
            if valid:
                # Check orthogonality with all existing squares
                for es in squares:
                    pairs = set()
                    for r in range(n):
                        for c in range(n):
                            p = (es[r][c], square[r][c])
                            if p in pairs:
                                valid = False
                                break
                            pairs.add(p)
                        if not valid:
                            break
                    if not valid:
                        break
            
            if valid:
                squares.append(square)
        
        if time.time() > deadline:
            break
    
    # Ensure we have at least 1 square
    if not squares:
        squares = [s0]
    
    # Convert to ensure Python ints and lists
    result = []
    for sq in squares:
        result.append([[int(v) for v in row] for row in sq])
    
    return result

# EVOLVE-BLOCK-END
