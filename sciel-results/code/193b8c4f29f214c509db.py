"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 23 decomposition.
    # Structure:
    # 1. Strassen for A[:, 0:2] * B[0:2, 0:2] (Rank 7)
    # 2. Schoolbook for A[:, 0:2] * B[0:2, 2:3] (Rank 4)
    # 3. Strassen for A[:, 0:2] * B[0:2, 3:5] (Rank 7)
    # 4. Schoolbook for A[:, 0:2] * B[0:2, 5:6] (Rank 4)
    # 5. Outer Product for A[:, 2] * B[2, :] (Rank 1)
    
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_cols, b_rows, b_cols, offset):
        # a_cols: [0, 1]
        # b_rows: [0, 1]
        # b_cols: [c0, c1]
        # offset: starting rank index
        # Strassen 7 terms
        # M1=(a00+a11)(b00+b11)
        # M2=(a10+a11)b00
        # M3=a00(b10-b11)
        # M4=a11(b01-b00)
        # M5=(a00+a01)b11
        # M6=(a10-a00)(b00+b01)
        # M7=(a01-a11)(b10+b11)
        
        # Mapping U: i*m + j
        # Mapping V: j*p + k
        # Mapping W: k*n + i
        
        c0, c1 = b_cols
        r0, r1 = b_rows
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        idx = offset
        # U
        U[0*m + 0][idx] += 1 # A00
        U[1*m + 1][idx] += 1 # A11
        # V
        V[r0*p + c0][idx] += 1 # B00
        V[r1*p + c1][idx] += 1 # B11
        # W
        W[c0*n + 0][idx] += 1 # C00
        W[c1*n + 1][idx] += 1 # C11
        
        # Term 2: (A10+A11)B00 -> C10, C11 (-)
        idx = offset + 1
        U[1*m + 0][idx] += 1
        U[1*m + 1][idx] += 1
        V[r0*p + c0][idx] += 1
        W[c0*n + 1][idx] += 1
        W[c1*n + 1][idx] += -1
        
        # Term 3: A00(B10-B11) -> C01, C11
        idx = offset + 2
        U[0*m + 0][idx] += 1
        V[r1*p + c0][idx] += 1
        V[r1*p + c1][idx] += -1
        W[c0*n + 0][idx] += 1
        W[c1*n + 1][idx] += 1
        
        # Term 4: A11(B01-B00) -> C00, C10
        idx = offset + 3
        U[1*m + 1][idx] += 1
        V[r0*p + c1][idx] += 1
        V[r0*p + c0][idx] += -1
        W[c0*n + 0][idx] += 1
        W[c0*n + 1][idx] += 1
        
        # Term 5: (A00+A01)B11 -> C00 (-), C01
        idx = offset + 4
        U[0*m + 0][idx] += 1
        U[0*m + 1][idx] += 1
        V[r1*p + c1][idx] += 1
        W[c0*n + 0][idx] += -1
        W[c1*n + 0][idx] += 1
        
        # Term 6: (A10-A00)(B00+B01) -> C11
        idx = offset + 5
        U[1*m + 0][idx] += 1
        U[0*m + 0][idx] += -1
        V[r0*p + c0][idx] += 1
        V[r0*p + c1][idx] += 1
        W[c1*n + 1][idx] += 1
        
        # Term 7: (A01-A11)(B10+B11) -> C00, C11
        idx = offset + 6
        U[0*m + 1][idx] += 1
        U[1*m + 1][idx] += -1
        V[r1*p + c0][idx] += 1
        V[r1*p + c1][idx] += 1
        W[c0*n + 0][idx] += 1
        W[c1*n + 1][idx] += 1

    def add_schoolbook_block(a_cols, b_rows, b_col, offset):
        # a_cols: [0, 1]
        # b_rows: [0, 1]
        # b_col: single col index
        # Rank 4: for each (j, i) pair? No, for each j in b_rows, i in a_rows?
        # C_ik = sum_j A_ij B_jk
        # For fixed k (b_col), we need sum_j A_ij B_jk.
        # We can use one term for each j?
        # Term j: U[i,j]=1, V[j,k]=1, W[k,i]=1.
        # Sum over j: sum_j 1 * A_ij * B_jk * 1.
        # This works.
        for j in b_rows:
            idx = offset + j
            for i in range(n):
                U[i*m + j][idx] += 1
            V[j*p + b_col][idx] += 1
            for i in range(n):
                W[b_col*n + i][idx] += 1

    # 1. Strassen for cols 0,1 of B, rows 0,1
    add_strassen_block([0,1], [0,1], [0,1], 0)
    
    # 2. Schoolbook for col 2 of B, rows 0,1
    # Uses ranks 7, 8, 9, 10 (4 ranks)
    add_schoolbook_block([0,1], [0,1], 2, 7)
    
    # 3. Strassen for cols 3,4 of B, rows 0,1
    add_strassen_block([0,1], [0,1], [3,4], 11)
    
    # 4. Schoolbook for col 5 of B, rows 0,1
    # Uses ranks 18, 19, 20, 21 (4 ranks)
    add_schoolbook_block([0,1], [0,1], 5, 18)
    
    # 5. Outer product for A col 2, B row 2
    idx = 22
    for i in range(n):
        U[i*m + 2][idx] += 1
    for k in range(p):
        V[2*p + k][idx] += 1
    for k in range(p):
        for i in range(n):
            W[k*n + i][idx] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
