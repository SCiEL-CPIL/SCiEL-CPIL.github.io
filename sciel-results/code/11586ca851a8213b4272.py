def construct():
    n, m, p = 2, 3, 10
    rank = 52
    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0] * rank for _ in range(p * n)]  # 20 x rank
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :] — rank 20
    # U indices: i*m + 0 = i*3 + 0
    # V indices: 0*p + k = k
    # W indices: k*n + i = 2k + i
    for i in range(n):
        for k in range(p):
            U[i * m][col] = 1
            V[k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col = 20
    
    # Part 2: A[:, 1:3] * B[1:9, :] — 4 blocks of 2 columns, each rank 7
    # A' rows: A00=i0,j1->idx1, A01=i0,j2->idx2, A10=i1,j1->idx4, A11=i1,j2->idx5
    # For block b, columns k0=2b+1, k0+1=2b+2
    # B00=(j1,k0)->V[1*p+k0], B01=(j1,k0+1)->V[1*p+k0+1]
    # B10=(j2,k0)->V[2*p+k0], B11=(j2,k0+1)->V[2*p+k0+1]
    # C00=(i0,k0)->W[2*k0], C01=(i0,k0+1)->W[2*k0+2]
    # C10=(i1,k0)->W[2*k0+1], C11=(i1,k0+1)->W[2*k0+3]
    
    for b in range(4):
        k0 = 2 * b + 1  # columns 1,2; 3,4; 5,6; 7,8
        
        # M1 = (A00 + A11)(B00 + B11)
        U[1][col] += 1
        U[5][col] += 1
        V[1 * p + k0][col] += 1
        V[2 * p + k0 + 1][col] += 1
        W[2 * k0][col] += 1           # C00
        W[2 * (k0 + 1) + 1][col] += 1 # C11
        col += 1
        
        # M2 = (A10 + A11)B00
        U[4][col] += 1
        U[5][col] += 1
        V[1 * p + k0][col] += 1
        W[2 * k0 + 1][col] += 1       # C10
        W[2 * (k0 + 1) + 1][col] -= 1 # C11
        col += 1
        
        # M3 = A00(B01 - B11)
        U[1][col] += 1
        V[1 * p + k0 + 1][col] += 1
        V[2 * p + k0 + 1][col] -= 1
        W[2 * (k0 + 1)][col] += 1     # C01
        W[2 * (k0 + 1) + 1][col] += 1 # C11
        col += 1
        
        # M4 = A11(B10 - B00)
        U[5][col] += 1
        V[2 * p + k0][col] += 1
        V[1 * p + k0][col] -= 1
        W[2 * k0][col] += 1           # C00
        W[2 * k0 + 1][col] += 1       # C10
        col += 1
        
        # M5 = (A00 + A01)B11
        U[1][col] += 1
        U[2][col] += 1
        V[2 * p + k0 + 1][col] += 1
        W[2 * k0][col] -= 1           # C00
        W[2 * (k0 + 1)][col] += 1     # C01
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[4][col] += 1
        U[1][col] -= 1
        V[1 * p + k0][col] += 1
        V[1 * p + k0 + 1][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1 # C11
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[2][col] += 1
        U[5][col] -= 1
        V[2 * p + k0][col] += 1
        V[2 * p + k0 + 1][col] += 1
        W[2 * k0][col] += 1           # C00
        col += 1
    
    # col = 20 + 28 = 48
    
    # Part 3: A[:, 1:3] * B[9, :] — rank 4
    # Column 9, rows j=1,2
    # C00=(i0,9)->W[18], C10=(i1,9)->W[19]
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][col] = 1
            V[j * p + 9][col] = 1
            W[9 * n + i][col] = 1
            col += 1
    
    # col = 52
    return {"U": U, "V": V, "W": W}
