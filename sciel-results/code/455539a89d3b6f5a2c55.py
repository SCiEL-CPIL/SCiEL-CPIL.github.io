"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank for 2x2 is 7.
    # We have p/2 = 5 blocks of 2 columns.
    # Total rank = 5 * 7 = 35.
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(p // 2):
        # b is the block index (0 to 4)
        # Columns in B are 2*b and 2*b+1
        # Rows in C are 0 and 1
        # Columns in C are 2*b and 2*b+1
        
        # Strassen terms
        # 1: (A00+A11)(B00+B11)
        U[0][col] = 1
        U[3][col] = 1
        V[2*b][col] = 1
        V[10 + 2*b + 1][col] = 1
        W[4*b][col] = 1
        W[4*b + 3][col] = 1
        col += 1
        
        # 2: (A10+A11)B00
        U[2][col] = 1
        U[3][col] = 1
        V[2*b][col] = 1
        W[4*b + 1][col] = 1
        W[4*b + 3][col] = -1
        col += 1
        
        # 3: A00(B01-B11)
        U[0][col] = 1
        V[2*b + 1][col] = 1
        V[10 + 2*b + 1][col] = -1
        W[4*b + 2][col] = 1
        W[4*b + 3][col] = 1
        col += 1
        
        # 4: A11(B10-B00)
        U[3][col] = 1
        V[10 + 2*b][col] = 1
        V[2*b][col] = -1
        W[4*b][col] = 1
        W[4*b + 1][col] = 1
        col += 1
        
        # 5: (A00+A01)B11
        U[0][col] = 1
        U[1][col] = 1
        V[10 + 2*b + 1][col] = 1
        W[4*b][col] = -1
        W[4*b + 2][col] = 1
        col += 1
        
        # 6: (A10-A00)(B00+B01)
        U[2][col] = 1
        U[0][col] = -1
        V[2*b][col] = 1
        V[2*b + 1][col] = 1
        W[4*b + 3][col] = 1
        col += 1
        
        # 7: (A01-A11)(B10+B11)
        U[1][col] = 1
        U[3][col] = -1
        V[10 + 2*b][col] = 1
        V[10 + 2*b + 1][col] = 1
        W[4*b][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
