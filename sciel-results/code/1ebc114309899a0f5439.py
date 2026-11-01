"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    rank = 22

    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    col = 0
    for q in range(3):
        # Strassen-like 7 terms for columns (2q, 2q+1)
        # Term 1: A_col * B_col
        a = [2 * q, 2 * q + 1]
        b = [6 + 2 * q, 6 + 2 * q + 1]
        c = [4 * q, 4 * q + 3]
        for r in a: U[r][col] += 1
        for r in b: V[r][col] += 1
        for r in c: W[r][col] += 1
        col += 1

        # Term 2: A_col * (B_col1 - B_col0)
        a = [2 * q]
        b = [6 + 2 * q + 1]
        for r in a: U[r][col] += 1
        V[6 + 2 * q + 1][col] += 1
        V[6 + 2 * q][col] -= 1
        # W: C21 = -C22? No, Strassen: C21 = M2 + M4. 
        # Let's stick to the derivation: 
        # M1 = (A0+A1)(B0+B1) -> C00, C11
        # M2 = (A0+A1)B0 -> C10, C11
        # M3 = A1(B1-B0) -> C01, C11
        # M4 = A0(B0+B1) -> C00, C01
        # M5 = (A1-A0)(B0+B1) -> C10, C11
        # M6 = (A1-A0)B1 -> C01, C10
        # M7 = A0B0 -> C00
        # My previous derivation was slightly different. Let's use standard Strassen.
        # M1: (A0+A1)(B0+B1). 
        # A: A0=1, A1=1. B: B0=1, B1=1.
        # W: C00 += M1, C11 += M1.
        # But wait, the previous response had a specific set of vectors.
        # Let's use the standard Strassen for 2x2.
        # M1 = (A0+A1)(B0+B1)
        # M2 = (A0+A1)B0
        # M3 = A1(B1-B0)
        # M4 = A0(B0+B1)
        # M5 = (A1-A0)(B0+B1)
        # M6 = (A1-A0)B1
        # M7 = A0B0
        
        # C00 = M1 - M2 + M4 - M7
        # C01 = M3 + M4
        # C10 = M2 - M5 + M6
        # C11 = M1 - M3 - M5
        
        # Let's rebuild the 7 terms properly.
        # Term 1: (A0+A1)(B0+B1)
        # Term 2: (A0+A1)B0
        # Term 3: A1(B1-B0)
        # Term 4: A0(B0+B1)
        # Term 5: (A1-A0)(B0+B1)
        # Term 6: (A1-A0)B1
        # Term 7: A0B0

        # Reset col to before term 2 and redo? No, col is already advanced.
        # Let's just implement the 7 terms for each block.
        pass
    
    # The above loop is messy. Let's restart the construction cleanly.
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    for q in range(3):
        A0_idx = 2 * q
        A1_idx = 2 * q + 1
        B0_idx = 6 + 2 * q
        B1_idx = 6 + 2 * q + 1
        C00_idx = 4 * q
        C01_idx = 4 * q + 1
        C10_idx = 4 * q + 2
        C11_idx = 4 * q + 3
        
        # M1 = (A0+A1)(B0+B1)
        U[A0_idx][col] += 1
        U[A1_idx][col] += 1
        V[B0_idx][col] += 1
        V[B1_idx][col] += 1
        W[C00_idx][col] += 1
        W[C11_idx][col] += 1
        col += 1
        
        # M2 = (A0+A1)B0
        U[A0_idx][col] += 1
        U[A1_idx][col] += 1
        V[B0_idx][col] += 1
        W[C10_idx][col] += 1
        W[C11_idx][col] -= 1
        col += 1
        
        # M3 = A1(B1-B0)
        U[A1_idx][col] += 1
        V[B1_idx][col] += 1
        V[B0_idx][col] -= 1
        W[C01_idx][col] += 1
        W[C11_idx][col] -= 1
        col += 1
        
        # M4 = A0(B0+B1)
        U[A0_idx][col] += 1
        V[B0_idx][col] += 1
        V[B1_idx][col] += 1
        W[C00_idx][col] += 1
        W[C01_idx][col] += 1
        col += 1
        
        # M5 = (A1-A0)(B0+B1)
        U[A1_idx][col] += 1
        U[A0_idx][col] -= 1
        V[B0_idx][col] += 1
        V[B1_idx][col] += 1
        W[C10_idx][col] -= 1
        W[C11_idx][col] -= 1
        col += 1
        
        # M6 = (A1-A0)B1
        U[A1_idx][col] += 1
        U[A0_idx][col] -= 1
        V[B1_idx][col] += 1
        W[C01_idx][col] += 1
        W[C10_idx][col] += 1
        col += 1
        
        # M7 = A0B0
        U[A0_idx][col] += 1
        V[B0_idx][col] += 1
        W[C00_idx][col] -= 1
        col += 1
        
    # Term 8: A_col2 * B_col2 (for j=2)
    # A: row 5 (i=1, j=2) and 2 (i=0, j=2)
    # B: row 12 (j=2, k=0) and 13 (j=2, k=1) ... wait, m=3, p=6.
    # j=2 means rows 12, 13, 14, 15, 16, 17 in V?
    # V index: j*p + k. j=2, p=6. Indices 12, 13, 14, 15, 16, 17.
    # A index: i*m + j. i=0, j=2 -> 2. i=1, j=2 -> 5.
    # C index: k*n + i. k=0..5, i=0,1.
    # We are multiplying A[:, 2] (2x1) by B[2, :] (1x6).
    # This is a rank-1 update? No, it's an outer product.
    # Rank of outer product of 2x1 and 1x6 is 1? No, it's a 2x6 matrix.
    # The tensor for this sub-problem is of rank 1?
    # C[i, k] = A[i, 2] * B[2, k].
    # This is separable. Rank 1.
    # U: A[i, 2] -> U[i*3+2, col] = 1 for i=0,1.
    # V: B[2, k] -> V[2*6+k, col] = 1 for k=0..5.
    # W: C[i, k] -> W[k*2+i, col] = 1 for all i,k.
    
    U[2][col] += 1
    U[5][col] += 1
    for k in range(6):
        V[12 + k][col] += 1
        W[2 * k + 0][col] += 1
        W[2 * k + 1][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
