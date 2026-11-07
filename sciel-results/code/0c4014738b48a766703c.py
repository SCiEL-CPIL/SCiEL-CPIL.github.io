import random
import time

def _generate_random_latin_square(n):
    """Generates a random Latin square of order n using backtracking."""
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    
    def backtrack(pos):
        if pos == n * n:
            return True
        
        # Heuristic: Find the cell with the fewest options (MRV)
        # For simplicity and speed in this small n, we can just iterate, 
        # but MRV is much faster for hard instances.
        # Since n=10 is small, simple backtracking with randomization might suffice, 
        # but let's use a simple MRV approximation or just random order.
        
        # To ensure we don't get stuck in a bad path, we pick a random empty cell
        # but prioritize those with fewer options.
        min_options = n
        best_cell = None
        best_candidates = []
        
        # We can just pick the next empty cell in row-major order for simplicity 
        # if we randomize the symbols.
        # Let's stick to row-major for S1 generation, it's fast enough.
        i, j = divmod(pos, n)
        
        # Randomize symbols to break symmetry
        symbols = list(range(n))
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

def _find_orthogonal_square(existing_squares, n, timeout=5.0):
    """Finds a Latin square orthogonal to all squares in existing_squares."""
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    # For each existing square k, track used pairs (val_k, val_new)
    pair_used = [set() for _ in existing_squares]
    
    start_time = time.time()
    
    # Cell ordering: We will use a simple recursive backtracking.
    # To improve performance, we can use MRV, but for n=10, 
    # a well-ordered backtracker is often sufficient.
    
    def backtrack(pos):
        if time.time() - start_time > timeout:
            return False
            
        if pos == n * n:
            return True
        
        i, j = divmod(pos, n)
        
        # Determine valid values
        # We can pre-calculate candidates or check on the fly.
        # Checking on the fly is O(k) where k is number of existing squares.
        
        # Heuristic: Try values in random order
        symbols = list(range(n))
        random.shuffle(symbols)
        
        for v in symbols:
            if v in row_used[i] or v in col_used[j]:
                continue
                
            # Check orthogonality
            conflict = False
            for k, sq in enumerate(existing_squares):
                pair = (sq[i][j], v)
                if pair in pair_used[k]:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Assign
            grid[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            for k, sq in enumerate(existing_squares):
                pair_used[k].add((sq[i][j], v))
                
            if backtrack(pos + 1):
                return True
            
            # Unassign
            grid[i][j] = -1
            row_used[i].remove(v)
            col_used[j].remove(v)
            for k, sq in enumerate(existing_squares):
                pair_used[k].remove((sq[i][j], v))
                
        return False

    if backtrack(0):
        return [row[:] for row in grid]
    return None

def construct():
    n = 10
    # Set a global time limit to ensure we return something
    global_limit = time.time() + 55.0
    
    # We want to maximize the number of squares.
    # Start with a random square, try to extend.
    # If we fail to find a second square quickly, try a new first square.
    
    best_squares = []
    
    while time.time() < global_limit:
        # Generate a random first square
        S1 = _generate_random_latin_square(n)
        if S1 is None:
            continue
            
        squares = [S1]
        
        # Try to find orthogonal squares
        # We attempt to find up to n-1 squares
        while len(squares) < n:
            remaining_time = global_limit - time.time()
            if remaining_time < 2.0:
                break
                
            # Use a short timeout for each step to allow restarts if stuck
            # For the first step (finding S2), we can allow a bit more time
            # but 5s is usually enough.
            timeout = min(5.0, remaining_time - 1.0)
            
            next_sq = _find_orthogonal_square(squares, n, timeout=timeout)
            
            if next_sq is None:
                break
                
            squares.append(next_sq)
            
        if len(squares) > len(best_squares):
            best_squares = squares
            
        # If we found a good set (e.g., 9), we can stop early?
        # The max is 9.
        if len(best_squares) == 9:
            break
            
    # If we found nothing better than 1, return the standard cyclic square
    # which is guaranteed to be valid (score 1.0)
    if not best_squares:
        best_squares = [[[(i+j)%n for j in range(n)] for i in range(n)]]
        
    # Ensure types are correct (Python int, list)
    # The generators already produce Python ints and lists.
    
    return best_squares
