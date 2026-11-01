"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 28 (4 blocks of Strassen rank 7)
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A indices: 00:0, 01:1, 10:2, 11:3
    # B indices for block b: 
    #   B00: 2b, B01: 2b+1
    #   B10: 8+2b, B11: 8+2b+1
    # C indices for block b:
    #   C00: 4b, C10: 4b+1
    #   C01: 4b+2, C11: 4b+3
    
    for b in range(4):
        # B indices
        v00 = 2 * b
        v01 = 2 * b + 1
        v10 = 8 + 2 * b
        v11 = 8 + 2 * b + 1
        
        # C indices
        w00 = 4 * b
        w10 = 4 * b + 1
        w01 = 4 * b + 2
        w11 = 4 * b + 3
        
        # Term 1: M1
        col = b * 7 + 0
        U[0][col] = 1
        U[3][col] = 1
        V[v00][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1
        W[w11][col] = -1
        
        # Term 2: M2
        col = b * 7 + 1
        U[2][col] = 1
        U[3][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        
        # Term 3: M3
        col = b * 7 + 2
        U[0][col] = 1
        V[v01][col] = 1
        V[v11][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        
        # Term 4: M4
        col = b * 7 + 3
        U[3][col] = 1
        V[v10][col] = 1
        V[v00][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        
        # Term 5: M5
        col = b * 7 + 4
        U[0][col] = 1
        U[1][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        
        # Term 6: M6
        col = b * 7 + 5
        U[2][col] = 1
        U[0][col] = -1
        V[v00][col] = 1
        V[v01][col] = 1
        W[w11][col] = 1
        
        # Term 7: M7
        col = b * 7 + 6
        U[1][col] = 1
        U[3][col] = -1
        V[v10][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
