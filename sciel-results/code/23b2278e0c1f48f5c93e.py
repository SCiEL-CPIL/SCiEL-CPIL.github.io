"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=4 into two 2-blocks, p=6 into three 2-blocks
    # Each sub-block is 2x2x2, decomposed via Strassen (rank 7)
    # Total rank = (m/2) * (p/2) * 7 = 2 * 3 * 7 = 42
    
    rank = (m // 2) * (p // 2) * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for a 2x2x2 block
    # U factors (A indices: A00=0, A01=1, A10=2, A11=3)
    U_strassen = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, -1, 0, 1],  # M7: A01-A11
    ]
    
    # V factors (B indices: B00=0, B01=1, B10=2, B11=3)
    V_strassen = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 0, 1, -1],  # M3: B10-B11
        [0, 1, -1, 0],  # M4: B01-B10
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B10
        [0, 1, 0, 1],   # M7: B01+B11
    ]
    
    # W factors (C indices: C00=0, C01=1, C10=2, C11=3)
    W_strassen = [
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [1, 1, 0, 0],   # M3: C00+C01
        [0, 0, 1, 1],   # M4: C10+C11
        [-1, 1, 0, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    column = 0
    for jb in range(m // 2):  # m-block index
        for kb in range(p // 2):  # p-block index
            for s in range(7):  # Strassen term
                # U: for each local a in {0,1}, j_local in {0,1}
                for a in range(2):
                    for j_local in range(2):
                        row_U = a * m + (jb * 2 + j_local)
                        U[row_U][column] = U_strassen[s][a * 2 + j_local]
                
                # V: for each local j_local in {0,1}, k_local in {0,1}
                for j_local in range(2):
                    for k_local in range(2):
                        row_V = (jb * 2 + j_local) * p + (kb * 2 + k_local)
                        V[row_V][column] = V_strassen[s][j_local * 2 + k_local]
                
                # W: for each local k_local in {0,1}, a in {0,1}
                # W index is k*n + i = (kb*2+k_local)*n + a
                for k_local in range(2):
                    for a in range(2):
                        row_W = (kb * 2 + k_local) * n + a
                        W[row_W][column] = W_strassen[s][k_local * 2 + a]
                
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
