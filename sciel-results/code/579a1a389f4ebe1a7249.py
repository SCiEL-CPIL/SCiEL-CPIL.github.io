"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Applied to three independent 2x2 blocks of the 2x6 matrix B
    # Block k corresponds to columns 2k and 2k+1 of B and C
    
    for k in range(3):
        base_col = 7 * k
        l_base = 2 * k
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00=1, A11=1
        U[0][base_col] = 1
        U[3][base_col] = 1
        # V: B00=1 (j=0, l=2k), B11=1 (j=1, l=2k+1)
        V[l_base][base_col] = 1
        V[6 + l_base + 1][base_col] = 1
        # W: C00=1 (l=2k, i=0), C11=1 (l=2k+1, i=1)
        W[l_base * 2 + 0][base_col] = 1
        W[(l_base + 1) * 2 + 1][base_col] = 1
        
        # M2 = (A10 + A11)B00
        # U: A10=1, A11=1
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        # V: B00=1
        V[l_base][base_col + 1] = 1
        # W: C10=1 (l=2k, i=1)
        W[l_base * 2 + 1][base_col + 1] = 1
        
        # M3 = A00(B01 - B11)
        # U: A00=1
        U[0][base_col + 2] = 1
        # V: B01=1 (j=0, l=2k+1), B11=-1 (j=1, l=2k+1)
        V[l_base + 1][base_col + 2] = 1
        V[6 + l_base + 1][base_col + 2] = -1
        # W: C01=1 (l=2k+1, i=0)
        W[(l_base + 1) * 2 + 0][base_col + 2] = 1
        
        # M4 = A11(B10 - B00)
        # U: A11=1
        U[3][base_col + 3] = 1
        # V: B10=1 (j=1, l=2k), B00=-1 (j=0, l=2k)
        V[6 + l_base][base_col + 3] = 1
        V[l_base][base_col + 3] = -1
        # W: C00=1 (l=2k, i=0), C10=1 (l=2k, i=1)
        W[l_base * 2 + 0][base_col + 3] = 1
        W[l_base * 2 + 1][base_col + 3] = 1
        
        # M5 = (A00 + A01)B11
        # U: A00=1, A01=1
        U[0][base_col + 4] = 1
        U[1][base_col + 4] = 1
        # V: B11=1
        V[6 + l_base + 1][base_col + 4] = 1
        # W: C00=-1 (l=2k, i=0), C01=1 (l=2k+1, i=0)
        W[l_base * 2 + 0][base_col + 4] = -1
        W[(l_base + 1) * 2 + 0][base_col + 4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # U: A10=1, A00=-1
        U[2][base_col + 5] = 1
        U[0][base_col + 5] = -1
        # V: B00=1 (j=0, l=2k), B01=1 (j=0, l=2k+1)
        V[l_base][base_col + 5] = 1
        V[l_base + 1][base_col + 5] = 1
        # W: C11=1 (l=2k+1, i=1)
        W[(l_base + 1) * 2 + 1][base_col + 5] = 1
        
        # M7 = (A01 - A11)(B10 + B11)
        # U: A01=1, A11=-1
        U[1][base_col + 6] = 1
        U[3][base_col + 6] = -1
        # V: B10=1 (j=1, l=2k), B11=1 (j=1, l=2k+1)
        V[6 + l_base][base_col + 6] = 1
        V[6 + l_base + 1][base_col + 6] = 1
        # W: C00=1 (l=2k, i=0)
        W[l_base * 2 + 0][base_col + 6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
