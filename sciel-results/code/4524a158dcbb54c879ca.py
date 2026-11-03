def _block_strassen_3x3_3x8():
    n, m, p = 3, 3, 8
    rank = 68
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    # 1. A_2x2 * B_2x8 using Strassen (4 times)
    # Strassen for 2x2 * 2x2
    # We need to map the 2x2 * 2x8 to 4 instances of 2x2 * 2x2
    # B_2x8 has columns 0..7
    # Split B into 4 blocks of 2 columns: B0 (cols 0,1), B1 (cols 2,3), B2 (cols 4,5), B3 (cols 6,7)
    
    # Strassen factors for 2x2 * 2x2
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    U_s, V_s, W_s = _strassen_2x2()
    
    for b_idx in range(4):
        # B block columns: 2*b_idx, 2*b_idx+1
        # C block columns: 2*b_idx, 2*b_idx+1
        # A block: rows 0,1; cols 0,1
        # W indices: k*n + i. k is col of C, i is row of C.
        # For this block, i in {0,1}, k in {2*b_idx, 2*b_idx+1}
        
        for c in range(7):
            # U: A[0,0], A[0,1], A[1,0], A[1,1]
            # Indices in U (3x3 flattened):
            # A[0,0] -> 0
            # A[0,1] -> 1
            # A[1,0] -> 3
            # A[1,1] -> 4
            U[0, col] = U_s[0, c]
            U[1, col] = U_s[1, c]
            U[3, col] = U_s[2, c]
            U[4, col] = U_s[3, c]
            
            # V: B[0, 2*b_idx], B[0, 2*b_idx+1], B[1, 2*b_idx], B[1, 2*b_idx+1]
            # Indices in V (3x8 flattened): j*p + k
            # B[0, 2*b_idx] -> 0*8 + 2*b_idx = 2*b_idx
            # B[0, 2*b_idx+1] -> 2*b_idx + 1
            # B[1, 2*b_idx] -> 1*8 + 2*b_idx = 8 + 2*b_idx
            # B[1, 2*b_idx+1] -> 8 + 2*b_idx + 1
            
            V[2*b_idx, col] = V_s[0, c]
            V[2*b_idx+1, col] = V_s[1, c]
            V[8 + 2*b_idx, col] = V_s[2, c]
            V[8 + 2*b_idx + 1, col] = V_s[3, c]
            
            # W: C[0, 2*b_idx], C[0, 2*b_idx+1], C[1, 2*b_idx], C[1, 2*b_idx+1]
            # Indices in W (8x3 flattened): k*3 + i
            # C[0, 2*b_idx] -> 2*b_idx*3 + 0 = 6*b_idx
            # C[0, 2*b_idx+1] -> (2*b_idx+1)*3 + 0 = 6*b_idx + 3
            # C[1, 2*b_idx] -> 2*b_idx*3 + 1 = 6*b_idx + 1
            # C[1, 2*b_idx+1] -> (2*b_idx+1)*3 + 1 = 6*b_idx + 4
            
            W[6*b_idx, col] = W_s[0, c]
            W[6*b_idx+3, col] = W_s[1, c]
            W[6*b_idx+1, col] = W_s[2, c]
            W[6*b_idx+4, col] = W_s[3, c]
            
            col += 1
            
    # 2. A_2x1 * B_1x8
    # A rows 0,1; col 2. B row 2; cols 0..7. C rows 0,1; cols 0..7.
    # Rank 2*8 = 16
    for i in range(2):
        for k in range(8):
            # A[i, 2] -> i*3 + 2
            U[i*3 + 2, col] = 1
            # B[2, k] -> 2*8 + k = 16 + k
            V[16 + k, col] = 1
            # C[i, k] -> k*3 + i
            W[k*3 + i, col] = 1
            col += 1

    # 3. A_1x2 * B_2x8
    # A row 2; cols 0,1. B rows 0,1; cols 0..7. C row 2; cols 0..7.
    # Rank 2*8 = 16
    for j in range(2):
        for k in range(8):
            # A[2, j] -> 2*3 + j = 6 + j
            U[6 + j, col] = 1
            # B[j, k] -> j*8 + k
            V[j*8 + k, col] = 1
            # C[2, k] -> k*3 + 2
            W[k*3 + 2, col] = 1
            col += 1

    # 4. A_1x1 * B_1x8
    # A[2,2]. B[2,k]. C[2,k].
    # Rank 1*8 = 8
    for k in range(8):
        U[8, col] = 1
        V[16 + k, col] = 1
        W[k*3 + 2, col] = 1
        col += 1

    return U, V, W
