import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Generate the first square (Cyclic)
    S1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [S1]
    
    # Function to check if a new square S_new is orthogonal to existing squares
    def is_orthogonal(S_new, S_old):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (S_old[i][j], S_new[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True
    
    # Function to find a new square orthogonal to a list of existing squares
    def find_square(existing_squares):
        # Backtracking search
        grid = [[-1 for _ in range(n)] for _ in range(n)]
        row_masks = [0] * n
        col_masks = [0] * n
        # For each existing square, we need to track used pairs
        # pair_masks[k] is a set of pairs used with existing_squares[k]
        pair_sets = [set() for _ in existing_squares]
        
        def solve(r, c):
            if time.time() > deadline:
                return False
            
            if r == n:
                return True
            
            if c == n:
                return solve(r + 1, 0)
            
            # Try all symbols 0..9
            # Optimization: Determine available symbols
            available = ~row_masks[r] & ~col_masks[c] & 0x3FF  # 10 bits
            
            # Check orthogonality constraints
            # For each existing square, the pair (S_old[r][c], symbol) must be new
            valid_symbols = []
            for s in range(n):
                if (available >> s) & 1:
                    ok = True
                    for k, S_old in enumerate(existing_squares):
                        p = (S_old[r][c], s)
                        if p in pair_sets[k]:
                            ok = False
                            break
                    if ok:
                        valid_symbols.append(s)
            
            # Randomize order for different search paths if needed
            random.shuffle(valid_symbols)
            
            for s in valid_symbols:
                # Place s
                grid[r][c] = s
                row_masks[r] |= (1 << s)
                col_masks[c] |= (1 << s)
                
                # Record pairs
                pairs_to_add = []
                for k, S_old in enumerate(existing_squares):
                    p = (S_old[r][c], s)
                    pair_sets[k].add(p)
                    pairs_to_add.append((k, p))
                
                if solve(r, c + 1):
                    return True
                
                # Backtrack
                for k, p in pairs_to_add:
                    pair_sets[k].remove(p)
                row_masks[r] &= ~(1 << s)
                col_masks[c] &= ~(1 << s)
                grid[r][c] = -1
                
            return False
        
        if solve(0, 0):
            return grid
        return None

    # Try to find up to 9 squares
    max_squares = 9
    attempts = 0
    while len(squares) < max_squares and time.time() < deadline:
        # Restart logic: if stuck, maybe randomize? 
        # For now, just try to find the next one.
        # If it fails, we might be stuck. 
        # To improve, we could randomize the order of symbols in valid_symbols.
        
        new_sq = find_square(squares)
        if new_sq:
            squares.append(new_sq)
        else:
            # If we can't find another, break. 
            # We could try different random seeds or strategies, but time is limited.
            break

    return squares

