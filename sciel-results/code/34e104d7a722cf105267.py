"""Matrix multiplication 6x6x6 - Strassen decomposition on 2x2 blocks of 3x3 matrices."""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen on 2x2 blocks: A, B, C are 6x6, split into 2x2 blocks of 3x3
    # A = [[A00, A01], [A10, A11]] where each block is 3x3
    # Similarly for B and C
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Rank = 7 * 3 * 3 * 3 = 189 (each Strassen multiply involves 3x3x3 = 27 scalar mults)
    rank = 189
    
    # Build U (n*m = 36 rows, rank cols), V (m*p = 36 rows, rank cols), W (p*n = 36 rows, rank cols)
    # U[i, j] where i = row of A (0..5), j = col of A (0..5) -> index i*m+j
    # V[j, k] where j = row of B (0..5), k = col of B (0..5) -> index j*p+k
    # W[k, i] where k = row of C (0..5), i = col of C (0..5) -> index k*n+i (trace convention)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for the 7 multiplications
    # For each t in 0..6, define x[t][bi][bj], y[t][bj][bk], c[t][bi][bk]
    # bi, bj, bk in {0, 1}
    
    # M1 = (A00+A11)(B00+B11)
    # x[0]: A00=1, A11=1 -> x[0][0][0]=1, x[0][1][1]=1
    # y[0]: B00=1, B11=1 -> y[0][0][0]=1, y[0][1][1]=1
    # c[0]: C00 gets +1, C11 gets +1 -> c[0][0][0]=1, c[0][1][1]=1
    
    # M2 = (A10+A11)B00
    # x[1]: A10=1, A11=1 -> x[1][1][0]=1, x[1][1][1]=1
    # y[1]: B00=1 -> y[1][0][0]=1
    # c[1]: C10 gets +1, C11 gets -1 -> c[1][1][0]=1, c[1][1][1]=-1
    
    # M3 = A00(B01-B11)
    # x[2]: A00=1 -> x[2][0][0]=1
    # y[2]: B01=1, B11=-1 -> y[2][0][1]=1, y[2][1][1]=-1
    # c[2]: C01 gets +1, C11 gets +1 -> c[2][0][1]=1, c[2][1][1]=1
    
    # M4 = A11(B10-B00)
    # x[3]: A11=1 -> x[3][1][1]=1
    # y[3]: B10=1, B00=-1 -> y[3][1][0]=1, y[3][0][0]=-1
    # c[3]: C00 gets +1, C10 gets +1 -> c[3][0][0]=1, c[3][1][0]=1
    
    # M5 = (A00+A01)B11
    # x[4]: A00=1, A01=1 -> x[4][0][0]=1, x[4][0][1]=1
    # y[4]: B11=1 -> y[4][1][1]=1
    # c[4]: C00 gets -1, C01 gets +1 -> c[4][0][0]=-1, c[4][0][1]=1
    
    # M6 = (A10-A00)(B00+B01)
    # x[5]: A10=1, A00=-1 -> x[5][1][0]=1, x[5][0][0]=-1
    # y[5]: B00=1, B01=1 -> y[5][0][0]=1, y[5][0][1]=1
    # c[5]: C11 gets +1 -> c[5][1][1]=1
    
    # M7 = (A01-A11)(B10+B11)
    # x[6]: A01=1, A11=-1 -> x[6][0][1]=1, x[6][1][1]=-1
    # y[6]: B10=1, B11=1 -> y[6][1][0]=1, y[6][1][1]=1
    # c[6]: C00 gets +1 -> c[6][0][0]=1
    
    x = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    
    y = [
        [[1, 0], [0, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 0], [1, 1]]   # M7
    ]
    
    c = [
        [[1, 0], [0, 1]],  # M1: C00+1, C11+1
        [[0, 0], [1, -1]], # M2: C10+1, C11-1
        [[0, 1], [0, 1]],  # M3: C01+1, C11+1
        [[1, 0], [1, 0]],  # M4: C00+1, C10+1
        [[-1, 1], [0, 0]], # M5: C00-1, C01+1
        [[0, 0], [0, 1]],  # M6: C11+1
        [[1, 0], [0, 0]]   # M7: C00+1
    ]
    
    # For each Strassen step t, and each (li, lj, lk) in 0..2 (within 3x3 blocks),
    # we have one scalar multiplication contributing to rank
    col = 0
    for t in range(7):
        for li in range(3):
            for lj in range(3):
                for lk in range(3):
                    # U contribution: A has blocks at (bi, bj)
                    for bi in range(2):
                        for bj in range(2):
                            val = x[t][bi][bj]
                            if val != 0:
                                # A row index: bi*3 + li, col index: bj*3 + lj
                                I = bi * 3 + li
                                J = bj * 3 + lj
                                U[I * m + J][col] = val
                    
                    # V contribution: B has blocks at (bj, bk)
                    for bj in range(2):
                        for bk in range(2):
                            val = y[t][bj][bk]
                            if val != 0:
                                # B row index: bj*3 + lj, col index: bk*3 + lk
                                J = bj * 3 + lj
                                K = bk * 3 + lk
                                V[J * p + K][col] = val
                    
                    # W contribution: C has blocks at (bk, bi) -> W index K*n + I
                    for bi in range(2):
                        for bk in range(2):
                            val = c[t][bi][bk]
                            if val != 0:
                                # C row index: bk*3 + lk, col index: bi*3 + li
                                K = bk * 3 + lk
                                I = bi * 3 + li
                                W[K * n + I][col] = val
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
