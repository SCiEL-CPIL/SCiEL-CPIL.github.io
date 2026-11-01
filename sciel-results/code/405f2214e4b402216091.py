"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11 decomposition:
    # 7 terms for Strassen 2x2x2 (columns 0,1 of B)
    # 4 terms for schoolbook 2x1x1 (column 2 of B)
    rank = 11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a term
    def set_term(u, v, w, r, val=1):
        U[u][r] += val
        V[v][r] += val
        W[w][r] += val

    # Strassen terms for B columns 0, 1
    # M1 = A11 * (B11 + B22)
    # u=0, v=0, v=4 (j=1, k=1 -> 1*3+1=4)
    # w for C11 (k=0, i=0 -> 0*2+0=0), C12 (k=1, i=0 -> 1*2+0=2)
    # Strassen formulas:
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M4
    # C21 = M1 + M2
    # C22 = M2 - M3 + M5 + M7
    
    # M1: A00 * (B00 + B11)
    # A00 is u=0.
    # B00 is v=0. B11 is v=4.
    # Output C00 (w=0) and C01 (w=2)? No, M1 contributes to C00 and C10.
    # Wait, Strassen M1 = A11(B11+B22).
    # C11 += M1. C21 += M1.
    # C11 is w=0. C21 is w=1 (k=0, i=1 -> 0*2+1=1).
    set_term(0, 0, 0, 0, 1)
    set_term(0, 0, 1, 0, 1) # M1 contributes to C21? No.
    # Let's map Strassen indices carefully.
    # A indices: 00, 01, 10, 11 -> u=0,1,2,3
    # B indices: 00, 01, 10, 11 -> v=0,1,3,4 (for k=0,1)
    # C indices: 00, 01, 10, 11 -> w=0,2,1,3 (for k=0,1)
    
    # M1 = A00 * (B00 + B11)
    # Contributes to C00 and C10.
    # C00: w=0. C10: w=1.
    # So term 0: U[0,0]=1, V[0,0]=1, V[4,0]=1, W[0,0]=1, W[1,0]=1.
    U[0][0] = 1
    V[0][0] = 1
    V[4][0] = 1
    W[0][0] = 1
    W[1][0] = 1

    # M2 = (A00 + A10) * B11
    # A00 u=0, A10 u=2. B11 v=4.
    # Contributes to C10 and C11.
    # C10: w=1. C11: w=3.
    U[0][1] = 1
    U[2][1] = 1
    V[4][1] = 1
    W[1][1] = 1
    W[3][1] = 1

    # M3 = (A01 + A11) * B00
    # A01 u=1, A11 u=3. B00 v=0.
    # Contributes to C01 and C11.
    # C01: w=2. C11: w=3.
    U[1][2] = 1
    U[3][2] = 1
    V[0][2] = 1
    W[2][2] = 1
    W[3][2] = 1

    # M4 = A11 * (B00 - B11)
    # A11 u=3. B00 v=0, B11 v=4.
    # Contributes to C00 and C01.
    # C00: w=0. C01: w=2.
    U[3][3] = 1
    V[0][3] = 1
    V[4][3] = -1
    W[0][3] = 1
    W[2][3] = 1

    # M5 = (A00 + A01) * B10
    # A00 u=0, A01 u=1. B10 v=3.
    # Contributes to C01 and C11.
    # C01: w=2. C11: w=3.
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1
    W[3][4] = 1

    # M6 = (A10 - A00) * (B00 + B01)
    # A10 u=2, A00 u=0. B00 v=0, B01 v=1.
    # Contributes to C00 and C01.
    # C00: w=0. C01: w=2.
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[0][5] = 1
    W[2][5] = 1

    # M7 = (A11 - A01) * (B10 + B11)
    # A11 u=3, A01 u=1. B10 v=3, B11 v=4.
    # Contributes to C00 and C10.
    # C00: w=0. C10: w=1.
    U[3][6] = 1
    U[1][6] = -1
    V[3][6] = 1
    V[4][6] = 1
    W[0][6] = 1
    W[1][6] = 1

    # Schoolbook for column 2 (k=2)
    # B indices: B02 v=2, B12 v=5.
    # C indices: C02 w=4, C12 w=5.
    
    # M8 = A00 * B02 -> C02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1

    # M9 = A01 * B12 -> C02
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1

    # M10 = A10 * B02 -> C12
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1

    # M11 = A11 * B12 -> C12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
