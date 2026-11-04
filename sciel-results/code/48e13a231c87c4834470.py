"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    total_rank = 0
    U = []
    V = []
    W = []

    def add_decomposition(u_local, v_local, w_local, i_offset, j_offset, k_offset, n_local, m_local, p_local):
        """
        Adds a local decomposition to the global U, V, W.
        u_local: list of lists, shape (n_local*m_local, rank_local)
        v_local: list of lists, shape (m_local*p_local, rank_local)
        w_local: list of lists, shape (p_local*n_local, rank_local)
        i_offset, j_offset, k_offset: starting indices for A, B, C blocks
        n_local, m_local, p_local: dimensions of the block
        """
        nonlocal total_rank
        rank_local = len(u_local[0]) if u_local else 0
        for c in range(rank_local):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Map U (A indices)
            for i in range(n_local):
                for j in range(m_local):
                    val = u_local[i * m_local + j][c]
                    if val != 0:
                        u_col[(i_offset + i) * m + (j_offset + j)] = val
                        
            # Map V (B indices)
            for j in range(m_local):
                for k in range(p_local):
                    val = v_local[j * p_local + k][c]
                    if val != 0:
                        v_col[(j_offset + j) * p + (k_offset + k)] = val
                        
            # Map W (C indices)
            for k in range(p_local):
                for i in range(n_local):
                    val = w_local[k * n_local + i][c]
                    if val != 0:
                        w_col[(k_offset + k) * n + (i_offset + i)] = val

            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
        total_rank += rank_local

    def schoolbook(n, m, p):
        """Returns U, V, W for schoolbook multiplication of n x m * m x p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2():
        """Returns U, V, W for Strassen multiplication of 2 x 2 * 2 x 2."""
        # Indices for A: 00, 01, 10, 11
        # Indices for B: 00, 01, 10, 11
        # Indices for C: 00, 01, 10, 11
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1
        U[0][0] = 1; U[3][0] = 1 # a, d
        V[0][0] = 1; V[3][0] = 1 # e, h
        W[0][0] = 1; W[3][0] = 1 # C00, C11 (coeff 1)
        
        # m2
        U[2][1] = 1; U[3][1] = 1 # c, d
        V[0][1] = 1 # e
        W[2][1] = 1 # C10 (coeff 1)
        
        # m3
        U[0][2] = 1 # a
        V[1][2] = 1; V[3][2] = -1 # f, -h
        W[1][2] = 1; W[3][2] = 1 # C01, C11 (coeff 1)
        
        # m4
        U[3][3] = 1 # d
        V[2][3] = 1; V[0][3] = -1 # g, -e
        W[0][3] = 1; W[2][3] = 1 # C00, C10 (coeff 1)
        
        # m5
        U[0][4] = 1; U[1][4] = 1 # a, b
        V[3][4] = 1 # h
        W[0][4] = -1; W[1][4] = 1 # C00 (-1), C01 (1)
        
        # m6
        U[2][5] = 1; U[0][5] = -1 # c, -a
        V[0][5] = 1; V[1][5] = 1 # e, f
        W[3][5] = 1 # C11 (coeff 1)
        
        # m7
        U[1][6] = 1; U[3][6] = -1 # b, -d
        V[2][6] = 1; V[3][6] = 1 # g, h
        W[0][6] = 1 # C00 (coeff 1)
        
        return U, V, W

    # Partition indices into two sets: {0, 1} and {2, 3, 4}
    # Set 1: 0, 1 (size 2)
    # Set 2: 2, 3, 4 (size 3)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # This is 2x2 * 2x2. Use Strassen.
    U1, V1, W1 = strassen_2x2()
    add_decomposition(U1, V1, W1, 0, 0, 0, 2, 2, 2)
    
    # Block 2: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # This is 2x3 * 3x2. Use Schoolbook.
    U2, V2, W2 = schoolbook(2, 3, 2)
    add_decomposition(U2, V2, W2, 0, 2, 0, 2, 3, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # This is 2x2 * 2x3. Use Schoolbook.
    U3, V3, W3 = schoolbook(2, 2, 3)
    add_decomposition(U3, V3, W3, 0, 0, 2, 2, 2, 3)
    
    # Block 4: A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # This is 2x3 * 3x3. Use Schoolbook.
    U4, V4, W4 = schoolbook(2, 3, 3)
    add_decomposition(U4, V4, W4, 0, 2, 2, 2, 3, 3)
    
    # Block 5: A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2]
    # This is 3x2 * 2x2. Use Schoolbook.
    U5, V5, W5 = schoolbook(3, 2, 2)
    add_decomposition(U5, V5, W5, 2, 0, 0, 3, 2, 2)
    
    # Block 6: A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2]
    # This is 3x3 * 3x2. Use Schoolbook.
    U6, V6, W6 = schoolbook(3, 3, 2)
    add_decomposition(U6, V6, W6, 2, 2, 0, 3, 3, 2)
    
    # Block 7: A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5]
    # This is 3x2 * 2x3. Use Schoolbook.
    U7, V7, W7 = schoolbook(3, 2, 3)
    add_decomposition(U7, V7, W7, 2, 0, 2, 3, 2, 3)
    
    # Block 8: A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5]
    # This is 3x3 * 3x3. Use Schoolbook.
    U8, V8, W8 = schoolbook(3, 3, 3)
    add_decomposition(U8, V8, W8, 2, 2, 2, 3, 3, 3)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
