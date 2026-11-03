# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    
    U = np.zeros((n * m, 100))  # 12 x 100, will trim
    V = np.zeros((m * p, 100))  # 28 x 100
    W = np.zeros((p * n, 100))  # 21 x 100
    
    rank = 0
    
    def add_strassen_2x2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """Add Strassen 2x2x2 decomposition."""
        nonlocal rank
        i1, i2 = A_rows
        j1, j2 = A_cols
        k1, k2 = B_rows
        l1, l2 = B_cols
        r1, r2 = C_rows
        s1, s2 = C_cols
        
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[k1,l1]+B[k2,l2])
        U[i1*m+j1, rank] += 1
        U[i2*m+j2, rank] += 1
        V[k1*p+l1, rank] += 1
        V[k2*p+l2, rank] += 1
        W[s1*n+r1, rank] += 1
        W[s2*n+r2, rank] += 1
        rank += 1
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[k1,l1]
        U[i2*m+j1, rank] += 1
        U[i2*m+j2, rank] += 1
        V[k1*p+l1, rank] += 1
        W[s1*n+r2, rank] += 1
        rank += 1
        
        # M3 = A[i1,j1] * (B[k1,l2]-B[k2,l2])
        U[i1*m+j1, rank] += 1
        V[k1*p+l2, rank] += 1
        V[k2*p+l2, rank] += -1
        W[s2*n+r1, rank] += 1
        W[s2*n+r2, rank] += 1
        rank += 1
        
        # M4 = A[i2,j2] * (B[k2,l1]-B[k1,l1])
        U[i2*m+j2, rank] += 1
        V[k2*p+l1, rank] += 1
        V[k1*p+l1, rank] += -1
        W[s1*n+r1, rank] += 1
        W[s1*n+r2, rank] += 1
        rank += 1
        
        # M5 = (A[i1,j1]+A[i1,j2]) * B[k2,l2]
        U[i1*m+j1, rank] += 1
        U[i1*m+j2, rank] += 1
        V[k2*p+l2, rank] += 1
        W[s1*n+r1, rank] += -1
        W[s2*n+r1, rank] += 1
        rank += 1
        
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[k1,l1]+B[k1,l2])
        U[i2*m+j1, rank] += 1
        U[i1*m+j1, rank] += -1
        V[k1*p+l1, rank] += 1
        V[k1*p+l2, rank] += 1
        W[s2*n+r2, rank] += 1
        rank += 1
        
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[k2,l1]+B[k2,l2])
        U[i1*m+j2, rank] += 1
        U[i2*m+j2, rank] += -1
        V[k2*p+l1, rank] += 1
        V[k2*p+l2, rank] += 1
        W[s1*n+r1, rank] += 1
        rank += 1
    
    def add_dot(A_row, A_col, B_row, B_col, C_row, C_col, coeff=1):
        """Add rank-1 term: coeff * A[A_row,A_col] * B[B_row,B_col] -> C[C_row,C_col]"""
        nonlocal rank
        U[A_row*m + A_col, rank] += coeff
        V[B_row*p + B_col, rank] += 1
        W[C_col*n + C_row, rank] += 1
        rank += 1
    
    # B col blocks: (0,1), (2,3), (4,5), (6,)
    b_col_blocks = [(0,1), (2,3), (4,5), (6,)]
    
    # A row 0 (1x4) * B col blocks
    for k1, k2 in b_col_blocks:
        if k2 == k1:  # 1 col: 1x4 * 4x1, rank 4
            for j in range(4):
                add_dot(0, j, j, k1, 0, k1)
        else:  # 2 cols: 1x4 * 4x2, rank 8
            for j in range(4):
                add_dot(0, j, j, k1, 0, k1)
                add_dot(0, j, j, k2, 0, k2)
    
    # A rows 1,2 (2x4) * B col blocks
    for k1, k2 in b_col_blocks:
        if k2 == k1:  # 1 col: 2x4 * 4x1, rank 8
            for j in range(4):
                add_dot(1, j, j, k1, 1, k1)
                add_dot(2, j, j, k1, 2, k1)
        else:  # 2 cols: 2x4 * 4x2, split middle dim 4 into 2+2
            # A[1:3, 0:2] * B[0:2, k1:k2+1] + A[1:3, 2:4] * B[2:4, k1:k2+1]
            add_strassen_2x2x2((1,2), (0,1), (0,1), (k1,k2), (1,2), (k1,k2))
            add_strassen_2x2x2((1,2), (2,3), (2,3), (k1,k2), (1,2), (k1,k2))
    
    # Trim
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    if _verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
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
