"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Laderman's algorithm for 4x4 x 4x4, rank 21
    # We'll use it for 3x4 x 4x4 (pad A to 4x4 with zero row)
    # Then split 4x8 into two 4x4 blocks
    
    # Define Laderman terms for 4x4 x 4x4
    # Each term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict (i,j) -> val, b_coeffs: dict (j,k) -> val, c_coeffs: dict (i,k) -> val
    # Using 0-indexed
    
    laderman_terms = []
    
    # P1 = (A00+A11)(B00+B11)
    laderman_terms.append({
        'a': {(0,0):1, (1,1):1},
        'b': {(0,0):1, (1,1):1},
        'c': {(0,0):1, (1,1):1}  # C00 += P1, C11 += P1
    })
    # P2 = (A00-A11)(B00-B11)
    laderman_terms.append({
        'a': {(0,0):1, (1,1):-1},
        'b': {(0,0):1, (1,1):-1},
        'c': {(0,0):1, (1,1):1}  # C00 += P2, C11 += P2... wait
    })
    
    # This is getting error-prone. Let me use a different approach.
    # I'll just verify the schoolbook for now and return rank 96.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
