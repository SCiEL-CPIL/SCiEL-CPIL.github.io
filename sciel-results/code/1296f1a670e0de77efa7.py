def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen coefficients
    # M1..M7
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M3
    # C10 = M2 + M4
    # C11 = M5 - M1 + M3 - M7
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # a_rows, a_cols are indices in A (2x4)
        # b_rows, b_cols are indices in B (4x4)
        # c_rows, c_cols are indices in C (2x4)
        
        # A is 2x2 block at [a_rows, a_cols]
        # B is 2x2 block at [b_rows, b_cols]
        # C is 2x2 block at [c_rows, c_cols]
        
        # Map local A indices (0,0), (0,1), (1,0), (1,1) to global U indices
        # U index = i*m + j
        u_00 = a_rows[0] * m + a_cols[0]
        u_01 = a_rows[0] * m + a_cols[1]
        u_10 = a_rows[1] * m + a_cols[0]
        u_11 = a_rows[1] * m + a_cols[1]
        
        # Map local B indices to global V indices
        # V index = j*p + k
        v_00 = b_rows[0] * p + b_cols[0]
        v_01 = b_rows[0] * p + b_cols[1]
        v_10 = b_rows[1] * p + b_cols[0]
        v_11 = b_rows[1] * p + b_cols[1]
        
        # Map local C indices to global W indices
        # W index = k*n + i
        w_00 = c_cols[0] * n + c_rows[0]
        w_01 = c_cols[1] * n + c_rows[0]
        w_10 = c_cols[0] * n + c_rows[1]
        w_11 = c_cols[1] * n + c_rows[1]
        
        # M1 = A00 * (B00 - B10)
        U[u_00][col] += 1
        V[v_00][col] += 1
        V[v_10][col] -= 1
        # C contributions for M1: C01 += M1
        W[w_01][col] += 1
        col += 1
        
        # M2 = (A00 + A10) * B10
        U[u_00][col] += 1
        U[u_10][col] += 1
        V[v_10][col] += 1
        # C contributions for M2: C00 -= M2, C10 += M2
        W[w_00][col] -= 1
        W[w_10][col] += 1
        col += 1
        
        # M3 = A01 * (B00 + B01)
        U[u_01][col] += 1
        V[v_00][col] += 1
        V[v_01][col] += 1
        # C contributions for M3: C01 += M3, C11 += M3
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M4 = A11 * (B10 + B11)
        U[u_11][col] += 1
        V[v_10][col] += 1
        V[v_11][col] += 1
        # C contributions for M4: C00 += M4, C10 += M4
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5 = (A00 + A11) * (B00 + B11)
        U[u_00][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        V[v_11][col] += 1
        # C contributions for M5: C00 += M5, C11 += M5
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[u_10][col] += 1
        U[u_00][col] -= 1
        V[v_00][col] += 1
        V[v_01][col] += 1
        # C contributions for M6: C00 += M6
        W[w_00][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[u_01][col] += 1
        U[u_11][col] -= 1
        V[v_10][col] += 1
        V[v_11][col] += 1
        # C contributions for M7: C11 -= M7
        W[w_11][col] -= 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 3: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    return {"U": U, "V": V, "W": W}
