def construct():
    n, m, p = 2, 5, 6
    rank = 48
    
    U = [[0]*rank for _ in range(n*m)]  # 10 x 48
    V = [[0]*rank for _ in range(m*p)]  # 30 x 48
    W = [[0]*rank for _ in range(p*n)]  # 12 x 48
    
    col = 0
    
    def add_strassen_2x2_2x2(a_row_off, a_col_off, b_row_off, b_col_off):
        nonlocal col
        # A' is 2x2 at A[a_row_off+i, a_col_off+j]
        # B' is 2x2 at B[b_row_off+j, b_col_off+k]
        # C' is 2x2 at C[a_row_off+i, b_col_off+k]
        
        # Strassen terms for 2x2 * 2x2
        # P1 = (A11)(B11 - B01)
        # P2 = (A00 + A11)(B00 + B11)
        # P3 = (A00 + A01)(B11)
        # P4 = (A10 - A00)(B00 + B01)
        # P5 = (A00 + A10)(B01)
        # P6 = (A11 - A10)(B00 + B01)
        # P7 = (A00 - A10)(B11)
        
        # C00 = P2 + P4 - P5 - P7
        # C01 = P1 + P5
        # C10 = P3 + P6
        # C11 = P2 - P3 + P1 + P7
        
        def u(i, j): return i*m + a_col_off + j
        def v(j, k): return (b_row_off + j)*p + b_col_off + k
        def w(k, i): return (b_col_off + k)*n + a_row_off + i
        
        # P1
        U[u(1,1)][col] += 1
        V[v(1,1)][col] += 1
        V[v(0,1)][col] -= 1
        W[w(0,0)][col] += 1 # C01 part? No, P1 contributes to C01 and C11?
        # Standard Strassen:
        # C01 = P1 + P5
        # C11 = P2 - P3 + P1 + P7
        # So P1 contributes +1 to C01 and +1 to C11.
        W[w(0,1)][col] += 1 # C01 is k=1, i=0?
        # Wait, C indices: i (row), k (col).
        # w(k, i) maps to C[i, k].
        # C01 is i=0, k=1. w(1, 0).
        # C11 is i=1, k=1. w(1, 1).
        W[w(1,0)][col] += 1
        W[w(1,1)][col] += 1
        col += 1
        
        # P2
        U[u(0,0)][col] += 1
        U[u(1,1)][col] += 1
        V[v(0,0)][col] += 1
        V[v(1,1)][col] += 1
        # C00 = P2...
        # C11 = P2...
        W[w(0,0)][col] += 1
        W[w(1,1)][col] += 1
        col += 1
        
        # P3
        U[u(0,0)][col] += 1
        U[u(0,1)][col] += 1
        V[v(1,1)][col] += 1
        # C10 = P3...
        W[w(0,1)][col] += 1
        col += 1
        
        # P4
        U[u(1,0)][col] += 1
        U[u(0,0)][col] -= 1
        V[v(0,0)][col] += 1
        V[v(0,1)][col] += 1
        # C00 = P4...
        W[w(0,0)][col] += 1
        col += 1
        
        # P5
        U[u(0,0)][col] += 1
        U[u(1,0)][col] += 1
        V[v(0,1)][col] += 1
        # C00 = -P5...
        # C01 = P5...
        W[w(0,0)][col] -= 1
        W[w(1,0)][col] += 1
        col += 1
        
        # P6
        U[u(1,1)][col] += 1
        U[u(1,0)][col] -= 1
        V[v(0,0)][col] += 1
        V[v(0,1)][col] += 1
        # C10 = P6...
        W[w(0,1)][col] += 1
        col += 1
        
        # P7
        U[u(0,0)][col] += 1
        U[u(1,0)][col] -= 1
        V[v(1,1)][col] += 1
        # C00 = -P7...
        # C11 = P7...
        W[w(0,0)][col] -= 1
        W[w(1,1)][col] += 1
        col += 1

    def add_direct_2x1_2x2(a_row_off, a_col, b_row, b_col_off):
        nonlocal col
        # A[i, a_col] * B[b_row, b_col_off+k] -> C[i, b_col_off+k]
        for i in range(2):
            for k in range(2):
                U[i*m + a_col][col] += 1
                V[b_row*p + b_col_off + k][col] += 1
                W[(b_col_off + k)*n + i][col] += 1
                col += 1

    # Block 0: B cols 0,1
    add_strassen_2x2_2x2(0, 0, 0, 0) # A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_2x2_2x2(0, 2, 2, 0) # A[0:2, 2:4] * B[2:4, 0:2]
    add_direct_2x1_2x2(0, 4, 4, 0)   # A[0:2, 4] * B[4, 0:2]
    
    # Block 1: B cols 2,3
    add_strassen_2x2_2x2(0, 0, 0, 2) # A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_2x2_2x2(0, 2, 2, 2) # A[0:2, 2:4] * B[2:4, 2:4]
    add_direct_2x1_2x2(0, 4, 4, 2)   # A[0:2, 4] * B[4, 2:4]
    
    # Block 2: B cols 4,5
    add_strassen_2x2_2x2(0, 0, 0, 4) # A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_2x2_2x2(0, 2, 2, 4) # A[0:2, 2:4] * B[2:4, 4:6]
    add_direct_2x1_2x2(0, 4, 4, 4)   # A[0:2, 4] * B[4, 4:6]
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
