DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_idx, b_idx, c_idx, offset):
        # a_idx: (a, b, c, d) indices in U (flat)
        # b_idx: (e, f, g, h) indices in V (flat)
        # c_idx: (c00, c01, c10, c11) indices in W (flat)
        # offset: current column in U,V,W
        
        # P1 = (a+d)(e+h)
        U[a_idx[0]][offset] += 1
        U[a_idx[3]][offset] += 1
        V[b_idx[0]][offset] += 1
        V[b_idx[3]][offset] += 1
        W[c_idx[0]][offset] += 1
        W[c_idx[3]][offset] += 1
        col_offset = offset
        
        # P2 = (c+d)e
        U[a_idx[2]][col_offset+1] += 1
        U[a_idx[3]][col_offset+1] += 1
        V[b_idx[0]][col_offset+1] += 1
        W[c_idx[2]][col_offset+1] += 1
        
        # P3 = a(f-h)
        U[a_idx[0]][col_offset+2] += 1
        V[b_idx[1]][col_offset+2] += 1
        V[b_idx[3]][col_offset+2] -= 1
        W[c_idx[1]][col_offset+2] += 1
        W[c_idx[3]][col_offset+2] += 1
        
        # P4 = d(g-e)
        U[a_idx[3]][col_offset+3] += 1
        V[b_idx[2]][col_offset+3] += 1
        V[b_idx[0]][col_offset+3] -= 1
        W[c_idx[0]][col_offset+3] += 1
        W[c_idx[2]][col_offset+3] += 1
        
        # P5 = (a+b)h
        U[a_idx[0]][col_offset+4] += 1
        U[a_idx[1]][col_offset+4] += 1
        V[b_idx[3]][col_offset+4] += 1
        W[c_idx[0]][col_offset+4] -= 1
        W[c_idx[1]][col_offset+4] += 1
        
        # P6 = (c-a)(e+f)
        U[a_idx[2]][col_offset+5] += 1
        U[a_idx[0]][col_offset+5] -= 1
        V[b_idx[0]][col_offset+5] += 1
        V[b_idx[1]][col_offset+5] += 1
        W[c_idx[3]][col_offset+5] += 1
        
        # P7 = (b-d)(g+h)
        U[a_idx[1]][col_offset+6] += 1
        U[a_idx[3]][col_offset+6] -= 1
        V[b_idx[2]][col_offset+6] += 1
        V[b_idx[3]][col_offset+6] += 1
        W[c_idx[0]][col_offset+6] += 1
        W[c_idx[3]][col_offset+6] -= 1
        
        return 7

    def add_schoolbook_2x2x1(a_idx, b_idx, c_idx, offset):
        # a_idx: (a, b, c, d)
        # b_idx: (e, f)
        # c_idx: (c0, c1)
        # C0 = a*e + b*f
        # C1 = c*e + d*f
        
        # T1 = a*e
        U[a_idx[0]][offset] += 1
        V[b_idx[0]][offset] += 1
        W[c_idx[0]][offset] += 1
        
        # T2 = b*f
        U[a_idx[1]][offset+1] += 1
        V[b_idx[1]][offset+1] += 1
        W[c_idx[0]][offset+1] += 1
        
        # T3 = c*e
        U[a_idx[2]][offset+2] += 1
        V[b_idx[0]][offset+2] += 1
        W[c_idx[1]][offset+2] += 1
        
        # T4 = d*f
        U[a_idx[3]][offset+3] += 1
        V[b_idx[1]][offset+3] += 1
        W[c_idx[1]][offset+3] += 1
        
        return 4

    # Block 1: A1 * B1a
    # A1: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0, 1, 4, 5 in U (2x4)
    # B1a: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0, 1, 5, 6 in V (4x5)
    # C1a: C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0, 1, 2, 3 in W (5x2)
    a1 = [0, 1, 4, 5]
    b1a = [0, 1, 5, 6]
    c1a = [0, 1, 2, 3]
    col += add_strassen(a1, b1a, c1a, col)

    # Block 2: A1 * B1b
    # B1b: B[0,2], B[0,3], B[1,2], B[1,3] -> indices 2, 3, 7, 8
    # C1b: C[0,2], C[0,3], C[1,2], C[1,3] -> indices 4, 5, 6, 7
    b1b = [2, 3, 7, 8]
    c1b = [4, 5, 6, 7]
    col += add_strassen(a1, b1b, c1b, col)

    # Block 3: A1 * B1c
    # B1c: B[0,4], B[1,4] -> indices 4, 9
    # C1c: C[0,4], C[1,4] -> indices 8, 9
    b1c = [4, 9]
    c1c = [8, 9]
    col += add_schoolbook_2x2x1(a1, b1c, c1c, col)

    # Block 4: A2 * B2a
    # A2: A[0,2], A[0,3], A[1,2], A[1,3] -> indices 2, 3, 6, 7
    # B2a: B[2,0], B[2,1], B[3,0], B[3,1] -> indices 10, 11, 15, 16
    # C2a: C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0, 1, 2, 3 (Same C indices as C1a? No, C is the sum)
    # Yes, C = C1 + C2. So we add to the same W indices.
    a2 = [2, 3, 6, 7]
    b2a = [10, 11, 15, 16]
    c2a = [0, 1, 2, 3]
    col += add_strassen(a2, b2a, c2a, col)

    # Block 5: A2 * B2b
    # B2b: B[2,2], B[2,3], B[3,2], B[3,3] -> indices 12, 13, 17, 18
    # C2b: C[0,2], C[0,3], C[1,2], C[1,3] -> indices 4, 5, 6, 7
    b2b = [12, 13, 17, 18]
    c2b = [4, 5, 6, 7]
    col += add_strassen(a2, b2b, c2b, col)

    # Block 6: A2 * B2c
    # B2c: B[2,4], B[3,4] -> indices 14, 19
    # C2c: C[0,4], C[1,4] -> indices 8, 9
    b2c = [14, 19]
    c2c = [8, 9]
    col += add_schoolbook_2x2x1(a2, b2c, c2c, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
