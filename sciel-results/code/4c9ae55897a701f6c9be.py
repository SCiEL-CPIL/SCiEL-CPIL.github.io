def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 130.0

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    U_t = []
    V_t = []
    W_t = []
    
    # Strassen on A[1:3, 0:2] * B[0:2, 0:2]
    strassen_decomp(1, 0, 0, 0, U_t, V_t, W_t)
    # Strassen on A[1:3, 0:2] * B[0:2, 2:4]
    strassen_decomp(1, 0, 0, 2, U_t, V_t, W_t)
    # Strassen on A[1:3, 0:2] * B[0:2, 4:6]
    strassen_decomp(1, 0, 0, 4, U_t, V_t, W_t)
    # Strassen on A[1:3, 0:2] * B[0:2, 6:8]
    strassen_decomp(1, 0, 0, 6, U_t, V_t, W_t)
    
    # Direct terms for row 0 of A (all j values)
    for j in range(3):
        for k in range(8):
            direct_term(0, j, k, U_t, V_t, W_t)
    
    # Direct term for A[2,2] * B[2,k]
    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    ...
