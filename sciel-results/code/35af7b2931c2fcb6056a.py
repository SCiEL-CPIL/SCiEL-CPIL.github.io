import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    n = 10
    squares = []
    
    # Construct 5 MOLS using the Z_2 x Z_5 structure
    # Index i = 5*a_i + b_i where a_i in {0,1}, b_i in {0,2,3,4}
    # L_k[i][j] = 5 * ((a_i + a_j) % 2) + ((b_i + k * b_j) % 5)
    
    for k in range(5):
        if time.time() > deadline:
            break
            
        sq = [[0] * n for _ in range(n)]
        for i in range(n):
            a_i = i // 5
            b_i = i % 5
            for j in range(n):
                a_j = j // 5
                b_j = j % 5
                sq[i][j] = 5 * ((a_i + a_j) % 2) + ((b_i + k * b_j) % 5)
        
        # Verify this is a Latin square
        if not _is_latin_square(sq):
            continue
        
        # Check orthogonality with all existing squares
        orth = True
        for existing in squares:
            if not _are_orthogonal(existing, sq):
                orth = False
                break
        
        if orth:
            squares.append(sq)
            
    # Ensure we return Python ints and lists
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        
    return result
