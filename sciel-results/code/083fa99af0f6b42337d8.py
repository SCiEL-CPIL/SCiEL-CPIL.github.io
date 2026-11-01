DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    rank = 6 * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen for 2x2 x 2x2
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C10, 2=C01, 3=C11 (W[k*n+i] so C00=k0i0, C10=k0i1, C01=k1i0, C11=k1i1)
    
    strassen_U = [
        # P1, P2, P3, P4, P5, P6, P7
        [1,  0,  1,  0,  1,  0,  0],  # A00
        [0,  0,  0,  0,  1,  0,  1],  # A01
        [0,  1,  0,  0,  0, -1, 0],  # A10
        [1,  1,  0,  1,  0,  0, -1],  # A11
    ]
    
    strassen_V = [
        # P1, P2, P3, P4, P5, P6, P7
        [1,  1,  0, -1, 0,  1,  0],  # B00
        [1,  0,  1,  0,  0,  1,  1],  # B01
        [1,  0,  0,  1,  0,  0,  1],  # B10
        [1,  0, -1, 0,  1,  0,  1],  # B11
    ]
    
    # C indices: 0=C00(i0k0), 1=C10(i1k0), 2=C01(i0k1), 3=C11(i1k1)
    strassen_W = [
        # P1, P2, P3, P4, P5, P6, P7
        [1,  0,  0,  1, -1, 0,  1],  # C00 (i=0,k=0)
        [0,  1,  0,  1,  0,  0,  0],  # C10 (i=1,k=0)
        [0,  0,  1,  0,  1,  0,  0],  # C01 (i=0,k=1)
        [1, -1, 1,  0,  0,  1,  0],  # C11 (i=1,k=1)
    ]
    
    col = 0
    for group in range(6):
        for s in range(7):
            # U: A indices 0-3
            for ai in range(4):
                U[ai][col] = strassen_U[ai][s]
            # V: B indices for this group
            for bi in range(4):
                j = bi // 2
                l_offset = bi % 2
                l = 2 * group + l_offset
                V[j * p + l][col] = strassen_V[bi][s]
            # W: C indices
            for ci in range(4):
                i = ci % 2  # row
                k = ci // 2  # column
                W[k * n + i][col] = strassen_W[ci][s]
            col += 1
    
    return {"U": U, "V": V, "W": W}
