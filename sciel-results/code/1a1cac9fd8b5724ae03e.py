import time

def _generate_random_latin_square(n):
    """Generates a random Latin square of order n using backtracking."""
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    
    def backtrack(pos):
        if pos == n * n:
            return True
        
        i, j = divmod(pos, n)
        
        # MRV heuristic: find the cell with fewest options
        # In row-major order for a Latin square, the number of options is 
        # n - len(row_used[i]) - len(col_used[j]) + overlap.
        # Since we fill row by row, the first cell in a new row has n - len(col_used) options.
        # Actually, simple row-major with random symbols is often fast enough for n=10.
        
        symbols = list(range(n))
        import random
        random.shuffle(symbols)
        
        for v in symbols:
            if v not in row_used[i] and v not in col_used[j]:
                grid[i][j] = v
                row_used[i].add(v)
                col_used[j].add(v)
                
                if backtrack(pos + 1):
                    return True
                
                grid[i][j] = -1
                row_used[i].remove(v)
                col_used[j].remove(v)
        
        return False

    if backtrack(0):
        return [row[:] for row in grid]
    return None

def _find_orthogonal_mate(s1, n, timeout=10.0):
    """Finds a Latin square orthogonal to s1 using strong backtracking."""
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pair_used = set()
    
    start_time = time.time()
    
    def backtrack(pos):
        if time.time() - start_time > timeout:
            return False
        
        if pos == n * n:
            return True
        
        i, j = divmod(pos, n)
        
        # Calculate candidates
        candidates = []
        for v in range(n):
            if v in row_used[i] or v in col_used[j]:
                continue
            if (s1[i][j], v) in pair_used:
                continue
            candidates.append(v)
            
        if not candidates:
            return False
            
        # Randomize to avoid bias
        import random
        random.shuffle(candidates)
        
        for v in candidates:
            grid[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            pair_used.add((s1[i][j], v))
            
            if backtrack(pos + 1):
                return True
            
            grid[i][j] = -1
            row_used[i].remove(v)
            col_used[j].remove(v)
            pair_used.remove((s1[i][j], v))
            
        return False

    if backtrack(0):
        return [row[:] for row in grid]
    return None

def construct():
    n = 10
    # Generate a random Latin square
    s1 = _generate_random_latin_square(n)
    if s1 is None:
        # Fallback to cyclic
        s1 = [[(i + j) % n for j in range(n)] for i in range(n)]
        
    squares = [s1]
    
    # Find an orthogonal mate
    s2 = _find_orthogonal_mate(s1, n, timeout=50.0)
    if s2 is not None:
        squares.append(s2)
        
    return squares
