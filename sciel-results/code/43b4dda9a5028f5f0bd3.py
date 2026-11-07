import time

def construct():
    n = 10
    deadline = 85.0  # Safety margin before 90s kill
    
    # CRT mapping: x in Z_10 -> (x mod 2, x mod 5)
    # Inverse: given (a mod 2, b mod 5), find x mod 10
    # x = a*5*(5^{-1} mod 2) + b*2*(2^{-1} mod 5) mod 10
    # 5^{-1} mod 2 = 1, 2^{-1} mod 5 = 3
    # x = a*5*1 + b*2*3 = 5a + 6b mod 10
    def crt(a, b):
        return (5 * a + 6 * b) % 10
    
    # MOLS of order 2: L2(i,j) = (i + j) % 2
    # MOLS of order 5: L5_k(i,j) = (i + k*j) % 5 for k = 1,2,3,4
    
    squares = []
    
    # Square 1: from MOLS(2)
    sq1 = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            i2, j2 = i % 2, j % 2
            i5, j5 = i % 5, j % 5
            val2 = (i2 + j2) % 2
            # For the "5-part", we can use any fixed valid Latin square pattern
            # Use val5 = 0 (constant) - but that won't be Latin!
            # Instead, we need a proper combination.
            # Actually, for the MOLS(2) component, the 5-part should be a fixed Latin square
            # Let's use val5 = (i5 + j5) % 5 as a base Latin square for the 5-part
            val5 = (i5 + j5) % 5
            sq1[i][j] = crt(val2, val5)
    squares.append(sq1)
    
    # Squares 2-5: from MOLS(5) with k = 1,2,3,4
    for k in range(1, 5):
        sq = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                i2, j2 = i % 2, j % 2
                i5, j5 = i % 5, j % 5
                # For the 2-part, use a fixed pattern: val2 = i2 (or j2, or i2+j2)
                # To ensure orthogonality between different k values, the 2-part 
                # must be the same for all these squares (since the 5-part differs)
                # Use val2 = (i2 + j2) % 2 (same as sq1's 2-part)
                val2 = (i2 + j2) % 2
                val5 = (i5 + k * j5) % 5
                sq[i][j] = crt(val2, val5)
        squares.append(sq)
    
    # Verify all squares are valid Latin squares
    def is_latin(sq):
        for row in sq:
            if sorted(row) != list(range(n)):
                return False
        for col in zip(*sq):
            if sorted(col) != list(range(n)):
                return False
        return True
    
    # Verify pairwise orthogonality
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == n * n
    
    # Check all squares
    valid_squares = []
    for sq in squares:
        if not is_latin(sq):
            continue
        # Check orthogonality with all previously added squares
        orthogonal_to_all = True
        for vsq in valid_squares:
            if not are_orthogonal(sq, vsq):
                orthogonal_to_all = False
                break
        if orthogonal_to_all:
            valid_squares.append(sq)
    
    # Convert to ensure Python int types
    result = []
    for sq in valid_squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    # Fallback: if something went wrong, return at least one valid square
    if not result:
        fallback = [[(i + j) % n for j in range(n)] for i in range(n)]
        result = [[int(fallback[i][j]) for j in range(n)] for i in range(n)]
    
    return result
