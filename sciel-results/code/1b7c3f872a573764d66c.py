"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 10
    
    # Known mutually orthogonal Latin squares of order 10
    # Using the standard construction from combinatorial design theory
    
    # Square 0: cyclic Latin square
    sq0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # For order 10, we can construct orthogonal mates using the following approach:
    # Use the fact that 10 = 2*5 and construct using group-based methods
    
    # Here's a verified set of MOLS(10) - these are known to be mutually orthogonal
    # Based on the construction using the dihedral group or similar
    
    # Let me use a simpler verified construction:
    # L_k[i][j] = (i + k*j) % 10 for k coprime to 10 won't work for orthogonality
    
    # Actually, for composite n, we need more careful construction.
    # Let me use known hardcoded MOLS from literature.
    
    # The following 2 squares are verified to be orthogonal MOLS of order 10
    sq1 = [
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
        [ 1,  2,  3,  4,  5,  6,  7,  8,  9,  0],
        [ 2,  3,  4,  5,  6,  7,  8,  9,  0,  1],
        [ 3,  4,  5,  6,  7,  8,  9,  0,  1,  2],
        [ 4,  5,  6,  7,  8,  9,  0,  1,  2,  3],
        [ 5,  6,  7,  8,  9,  0,  1,  2,  3,  4],
        [ 6,  7,  8,  9,  0,  1,  2,  3,  4,  5],
        [ 7,  8,  9,  0,  1,  2,  3,  4,  5,  6],
        [ 8,  9,  0,  1,  2,  3,  4,  5,  6,  7],
        [ 9,  0,  1,  2,  3,  4,  5,  6,  7,  8],
    ]
    # This is just sq0 again! Not orthogonal.
    
    # Let me properly construct orthogonal mates
    # For n=10, use the following known orthogonal pair:
    
    # L1[i][j] = (i + j) % 10
    # L2[i][j] = (i + 3*j) % 10 - but gcd(3,10)=1, need to check orthogonality
    
    # Check: pairs (L1[i][j], L2[i][j]) = ((i+j)%10, (i+3j)%10)
    # For fixed pair (a,b), we need i+j ≡ a and i+3j ≡ b (mod 10)
    # Subtract: 2j ≡ b-a (mod 10), which has solutions only if gcd(2,10)|b-a
    # So not all pairs are covered - not orthogonal.
    
    # For composite n, linear constructions don't give orthogonal pairs easily.
    # I'll use a backtracking search to find valid MOLS.
    
    import time
    start_time = time.time()
    deadline = start_time + 60  # 60 second deadline with safety margin
    
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            for j in range(n):
                col = [sq[r][j] for r in range(n)]
                if sorted(col) != list(range(n)):
                    return False
        return True
    
    def is_orthogonal(s1, s2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (s1[i][j], s2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True
    
    # Start with the trivial cyclic square
    base_sq = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def find_orthogonal_mate(existing_squares, timeout=30):
        """Find a Latin square orthogonal to all existing squares."""
        import random
        from itertools import permutations
        
        start = time.time()
        
        # Use backtracking with constraint propagation
        # Variables: each cell (i,j) of the new square
        # Constraints: row/col permutation, orthogonality with each existing square
        
        # For each cell, track possible values
        possible = [[set(range(n)) for _ in range(n)] for _ in range(n)]
        
        # Apply orthogonality constraints: for each existing square es,
        # at cell (i,j), value v is forbidden if (es[i][j], v) already "used"
        # But we don't know v yet, so this is a global constraint.
        
        # Simpler: use random restart with backtracking
        for attempt in range(1000):
            if time.time() > start + timeout:
                return None
            
            # Random initial assignment
            grid = [[-1] * n for _ in range(n)]
            for i in range(n):
                row_vals = list(range(n))
                random.shuffle(row_vals)
                for j in range(n):
                    grid[i][j] = row_vals[j]
            
            # Check if valid
            if not is_latin(grid):
                continue
            
            ortho_ok = True
            for es in existing_squares:
                if not is_orthogonal(grid, es):
                    ortho_ok = False
                    break
            
            if ortho_ok:
                return grid
        
        return None
    
    squares = [base_sq]
    
    # Try to find orthogonal mates
    for _ in range(8):  # Try to find up to 8 more
        if time.time() > deadline:
            break
        
        mate = find_orthogonal_mate(squares, timeout=5)
        if mate is not None:
            squares.append(mate)
    
    # Final verification
    def validate_all(squares):
        for sq in squares:
            if not is_latin(sq):
                return False
        for i in range(len(squares)):
            for j in range(i+1, len(squares)):
                if not is_orthogonal(squares[i], squares[j]):
                    return False
        return True
    
    if not validate_all(squares):
        squares = [base_sq]
    
    # Convert to pure Python ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return result

# EVOLVE-BLOCK-END
