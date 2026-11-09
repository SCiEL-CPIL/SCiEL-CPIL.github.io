"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 10}

def construct():
    n = P['n']
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Start with the standard Latin square: L[i][j] = (i + j) % n
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # We'll try to find additional squares orthogonal to all previously found ones
    squares = [base]
    
    # For each existing square, record the set of (symbol_a, symbol_b) pairs
    # For a new square to be orthogonal to all, its pairs with each must be unique
    
    def get_pairs(s1, s2):
        """Get all ordered pairs from two squares."""
        pairs = set()
        for i in range(n):
            for j in range(n):
                pairs.add((s1[i][j], s2[i][j]))
        return pairs
    
    def is_orthogonal(s1, s2):
        """Check if two squares are orthogonal (all 100 pairs distinct)."""
        pairs = get_pairs(s1, s2)
        return len(pairs) == n * n
    
    def backtrack_fill(square, row, col, row_used, col_used, forbidden_pairs, squares_so_far):
        """
        Backtracking search to fill a Latin square orthogonal to all in squares_so_far.
        
        square: the square being filled (list of lists, partially filled with -1 for empty)
        row: current row to fill
        col: current column to fill
        row_used: list of sets, row_used[i] = set of symbols used in row i
        col_used: list of sets, col_used[j] = set of symbols used in column j
        forbidden_pairs: for each existing square k, the set of (base_val, new_val) pairs already used
                         forbidden_pairs[k] = set of (squares_so_far[k][i][j], square[i][j]) for filled cells
        squares_so_far: list of completed squares that must be orthogonal to the new one
        """
        if time.time() > deadline:
            return False
        
        # Find next empty cell
        while row < n:
            if col < n and square[row][col] == -1:
                break
            col += 1
            if col >= n:
                row += 1
                col = 0
        if row >= n:
            return True  # All cells filled successfully
        
        # Try each symbol 0..n-1
        for sym in range(n):
            if sym in row_used[row] or sym in col_used[col]:
                continue
            
            # Check orthogonality constraints
            valid = True
            for k, sq in enumerate(squares_so_far):
                pair = (sq[row][col], sym)
                if pair in forbidden_pairs[k]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Place the symbol
            square[row][col] = sym
            row_used[row].add(sym)
            col_used[col].add(sym)
            for k in range(len(squares_so_far)):
                forbidden_pairs[k].add((squares_so_far[k][row][col], sym))
            
            # Recurse
            if backtrack_fill(square, row, col + 1, row_used, col_used, forbidden_pairs, squares_so_far):
                return True
            
            # Undo
            square[row][col] = -1
            row_used[row].remove(sym)
            col_used[col].remove(sym)
            for k in range(len(squares_so_far)):
                forbidden_pairs[k].discard((squares_so_far[k][row][col], sym))
        
        return False
    
    # Try to find a second square orthogonal to the base
    # Use a randomized approach with backtracking
    random.seed(42)
    
    max_attempts = 50
    found_second = False
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Initialize an empty square
        square = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        forbidden_pairs = [set() for _ in range(len(squares))]
        
        # Use randomized variable ordering: fill cells in random order
        # But we need to maintain row/col constraints, so let's do row-by-row
        # with random symbol ordering
        
        # Actually, let's use a smarter approach: fill row by row, but shuffle
        # the order of symbols to try
        
        def backtrack_fill_row(row_idx):
            if time.time() > deadline:
                return False
            
            if row_idx >= n:
                return True
            
            # Try to fill row row_idx
            # The symbols to use in this row are 0..n-1, but we need to pick
            # which goes in which column
            available_syms = list(range(n))
            random.shuffle(available_syms)
            
            # Use backtracking within the row
            def fill_row_col(col_idx, remaining_syms):
                if time.time() > deadline:
                    return False
                
                if col_idx >= n:
                    return True
                
                for idx, sym in enumerate(remaining_syms):
                    if sym in col_used[col_idx]:
                        continue
                    
                    # Check orthogonality
                    valid = True
                    for k, sq in enumerate(squares):
                        pair = (sq[row_idx][col_idx], sym)
                        if pair in forbidden_pairs[k]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Place
                    square[row_idx][col_idx] = sym
                    col_used[col_idx].add(sym)
                    for k in range(len(squares)):
                        forbidden_pairs[k].add((squares[k][row_idx][col_idx], sym))
                    
                    new_remaining = remaining_syms[:idx] + remaining_syms[idx+1:]
                    if fill_row_col(col_idx + 1, new_remaining):
                        return True
                    
                    # Undo
                    square[row_idx][col_idx] = -1
                    col_used[col_idx].remove(sym)
                    for k in range(len(squares)):
                        forbidden_pairs[k].discard((squares[k][row_idx][col_idx], sym))
                
                return False
            
            return fill_row_col(0, available_syms)
        
        if backtrack_fill_row(0):
            # Verify orthogonality
            orthogonal = all(is_orthogonal(base, square) for base in squares)
            if orthogonal:
                squares.append(square)
                found_second = True
                break
        else:
            continue
    
    # Try to find a third square (unlikely to succeed for order 10)
    if time.time() < deadline - 10:
        max_attempts_third = 10
        found_third = False
        
        for attempt in range(max_attempts_third):
            if time.time() > deadline:
                break
            
            square = [[-1 for _ in range(n)] for _ in range(n)]
            row_used = [set() for _ in range(n)]
            col_used = [set() for _ in range(n)]
            forbidden_pairs = [set() for _ in range(len(squares))]
            
            def backtrack_fill_row_third(row_idx):
                if time.time() > deadline:
                    return False
                
                if row_idx >= n:
                    return True
                
                available_syms = list(range(n))
                random.shuffle(available_syms)
                
                def fill_row_col_third(col_idx, remaining_syms):
                    if time.time() > deadline:
                        return False
                    
                    if col_idx >= n:
                        return True
                    
                    for idx, sym in enumerate(remaining_syms):
                        if sym in col_used[col_idx]:
                            continue
                        
                        valid = True
                        for k, sq in enumerate(squares):
                            pair = (sq[row_idx][col_idx], sym)
                            if pair in forbidden_pairs[k]:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        square[row_idx][col_idx] = sym
                        col_used[col_idx].add(sym)
                        for k in range(len(squares)):
                            forbidden_pairs[k].add((squares[k][row_idx][col_idx], sym))
                        
                        new_remaining = remaining_syms[:idx] + remaining_syms[idx+1:]
                        if fill_row_col_third(col_idx + 1, new_remaining):
                            return True
                        
                        square[row_idx][col_idx] = -1
                        col_used[col_idx].remove(sym)
                        for k in range(len(squares)):
                            forbidden_pairs[k].discard((squares[k][row_idx][col_idx], sym))
                    
                    return False
                
                return fill_row_col_third(0, available_syms)
            
            if backtrack_fill_row_third(0):
                orthogonal = all(is_orthogonal(s, square) for s in squares)
                if orthogonal:
                    squares.append(square)
                    found_third = True
                    break
            else:
                continue
    
    # Ensure all squares are valid lists of lists of ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return result
# EVOLVE-BLOCK-END
