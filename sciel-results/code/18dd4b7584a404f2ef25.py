import time

def construct():
    n = 10
    start_time = time.time()
    timeout = 55.0
    
    # Generate first square (cyclic)
    squares = []
    sq0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares.append(sq0)
    
    # Try to find more squares using backtracking with constraint propagation
    for _ in range(8):  # Max 9 total
        if time.time() - start_time > timeout:
            break
        
        next_sq = find_next_square(n, squares, start_time, timeout)
        if next_sq is not None:
            squares.append(next_sq)
    
    return squares

def find_next_square(n, existing_squares, start_time, timeout):
    """
    Find a Latin square orthogonal to all existing squares using
    backtracking with MRV heuristic.
    """
    num_existing = len(existing_squares)
    
    row_used = [0] * n
    col_used = [0] * n
    grid = [[-1] * n for _ in range(n)]
    
    # used_with_v[k][v] = set of existing_vals that have been paired with new_val v in square k
    used_with_v = [[set() for _ in range(n)] for _ in range(num_existing)]
    
    def count_valid_values(i, j):
        """Count how many values are valid for cell (i,j) given current state."""
        count = 0
        for v in range(n):
            bit = 1 << v
            if (row_used[i] & bit) or (col_used[j] & bit):
                continue
            valid = True
            for k in range(num_existing):
                ev = existing_squares[k][i][j]
                if ev in used_with_v[k][v]:
                    valid = False
                    break
            if valid:
                count += 1
        return count
    
    def backtrack():
        if time.time() - start_time > timeout:
            return False
        
        # Find unassigned cell with minimum valid values (MRV heuristic)
        min_count = n + 1
        best_i, best_j = -1, -1
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == -1:
                    c = count_valid_values(i, j)
                    if c == 0:
                        return False  # Dead end detected
                    if c < min_count:
                        min_count = c
                        best_i, best_j = i, j
                        if c == 1:
                            break
            if best_i != -1 and min_count == 1:
                break
        
        if best_i == -1:
            return True  # All cells assigned
        
        i, j = best_i, best_j
        
        # Try each valid value
        for v in range(n):
            bit = 1 << v
            if (row_used[i] & bit) or (col_used[j] & bit):
                continue
            
            valid = True
            for k in range(num_existing):
                ev = existing_squares[k][i][j]
                if ev in used_with_v[k][v]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Place value
            grid[i][j] = v
            row_used[i] |= bit
            col_used[j] |= bit
            
            for k in range(num_existing):
                ev = existing_squares[k][i][j]
                used_with_v[k][v].add(ev)
            
            if backtrack():
                return True
            
            # Undo
            grid[i][j] = -1
            row_used[i] &= ~bit
            col_used[j] &= ~bit
            for k in range(num_existing):
                ev = existing_squares[k][i][j]
                used_with_v[k][v].discard(ev)
        
        return False
    
    if backtrack():
        return grid
    return None
