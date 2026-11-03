def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U indices:
    # 0: a11, 1: a12, 2: a13
    # 3: a21, 4: a22, 5: a23
    
    # V indices:
    # 0-7: b11..b18 (row 1 of B)
    # 8-15: b21..b28 (row 2 of B)
    # 16-23: b31..b38 (row 3 of B)
    
    # W indices:
    # k*n + i
    # 0,2,4,6,8,10,12,14: C11, C13, C15, C17, C21, C23, C25, C27 ? No.
    # W[row, col]. row = k*n + i.
    # k=0 (col 0): i=0 -> 0, i=1 -> 1.
    # k=1 (col 1): i=0 -> 2, i=1 -> 3.
    # ...
    # k=7 (col 7): i=0 -> 14, i=1 -> 15.
    
    r = 0
    
    # Helper to add Strassen block
    def add_strassen_block(u_indices, v_indices, w_indices):
        nonlocal r
        # u_indices: [a, b, c, d] corresponding to A11, A12, A21, A22
        # v_indices: [e, f, g, h] corresponding to B11, B12, B21, B22
        # w_indices: [c11, c12, c21, c22] corresponding to C11, C12, C21, C22
        
        # M1 = (a+d)(e+h)
        U[u_indices[0]][r] = 1
        U[u_indices[3]][r] = 1
        V[v_indices[0]][r] = 1
        V[v_indices[2]][r] = 1
        W[w_indices[0]][r] = 1  # C11 += M1
        W[w_indices[3]][r] = 1  # C22 += M1
        r += 1
        
        # M2 = (c+d)e
        U[u_indices[2]][r] = 1
        U[u_indices[3]][r] = 1
        V[v_indices[0]][r] = 1
        W[w_indices[2]][r] = 1  # C21 += M2
        W[w_indices[3]][r] = -1 # C22 -= M2
        r += 1
        
        # M3 = a(f-h)
        U[u_indices[0]][r] = 1
        V[v_indices[1]][r] = 1
        V[v_indices[3]][r] = -1
        W[w_indices[1]][r] = 1  # C12 += M3
        W[w_indices[3]][r] = 1  # C22 += M3
        r += 1
        
        # M4 = d(g-e)
        U[u_indices[3]][r] = 1
        V[v_indices[2]][r] = 1
        V[v_indices[0]][r] = -1
        W[w_indices[0]][r] = 1  # C11 += M4
        W[w_indices[2]][r] = 1  # C21 += M4
        r += 1
        
        # M5 = (a+b)h
        U[u_indices[0]][r] = 1
        U[u_indices[1]][r] = 1
        V[v_indices[3]][r] = 1
        W[w_indices[0]][r] = -1 # C11 -= M5
        W[w_indices[1]][r] = 1  # C12 += M5
        r += 1
        
        # M6 = (c-a)(e+f)
        U[u_indices[2]][r] = 1
        U[u_indices[0]][r] = -1
        V[v_indices[0]][r] = 1
        V[v_indices[1]][r] = 1
        W[w_indices[3]][r] = 1  # C22 += M6
        r += 1
        
        # M7 = (b-d)(g+h)
        U[u_indices[1]][r] = 1
        U[u_indices[3]][r] = -1
        V[v_indices[2]][r] = 1
        V[v_indices[3]][r] = 1
        W[w_indices[0]][r] = 1  # C11 += M7
        r += 1

    # A1 indices: 0(a11), 1(a12), 3(a21), 4(a22)
    u_a1 = [0, 1, 3, 4]
    
    # Block 1: B cols 0,1
    # B11=b11(0), B12=b12(1), B21=b21(8), B22=b22(9)
    v_b1 = [0, 1, 8, 9]
    # C11(0), C12(2), C21(1), C22(3) -- Wait, W index mapping:
    # C11 is i=0, k=0 -> idx 0
    # C12 is i=0, k=1 -> idx 2
    # C21 is i=1, k=0 -> idx 1
    # C22 is i=1, k=1 -> idx 3
    w_c1 = [0, 2, 1, 3]
    add_strassen_block(u_a1, v_b1, w_c1)
    
    # Block 2: B cols 2,3
    v_b2 = [2, 3, 10, 11]
    w_c2 = [4, 6, 5, 7] # C13(4), C14(6), C23(5), C24(7)
    add_strassen_block(u_a1, v_b2, w_c2)
    
    # Block 3: B cols 4,5
    v_b3 = [4, 5, 12, 13]
    w_c3 = [8, 10, 9, 11] # C15(8), C16(10), C25(9), C26(11)
    add_strassen_block(u_a1, v_b3, w_c3)
    
    # Block 4: B cols 6,7
    v_b4 = [6, 7, 14, 15]
    w_c4 = [12, 14, 13, 15] # C17(12), C18(14), C27(13), C28(15)
    add_strassen_block(u_a1, v_b4, w_c4)
    
    # Part 2: A2 * Bbot
    # A2: a13(2), a23(5)
    # Bbot: b31..b38 (16..23)
    for k in range(8):
        # C1,k += a13 * b3,k+1
        # C1,k index: k*2 + 0
        U[2][r] = 1
        V[16 + k][r] = 1
        W[k * 2][r] = 1
        r += 1
        
        # C2,k += a23 * b3,k+1
        # C2,k index: k*2 + 1
        U[5][r] = 1
        V[16 + k][r] = 1
        W[k * 2 + 1][r] = 1
        r += 1
        
    return {"U": U, "V": V, "W": W}
