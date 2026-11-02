"""Matrix multiplication 2,2,13 - Strassen-based rank-46 decomposition."""

def construct():
    """Construct rank-46 tensor decomposition for (2x2)*(2x13) matrix multiplication.
    
    Splits B into 6 blocks of 2 columns (Strassen, rank 7 each) + 1 column (schoolbook, rank 4).
    Total rank = 6*7 + 4 = 46.
    """
    n, m, p = 2, 2, 13
    rank = 46
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 46
    V = [[0.0] * rank for _ in range(m * p)]   # 26 x 46
    W = [[0.0] * rank for _ in range(p * n)]   # 26 x 46

    # Strassen for (a,b;c,d)*(e,f;g,h):
    # M1=(a+d)(e+h), M2=(c+d)f, M3=a(g-h), M4=d(e-f), M5=(a+b)h, M6=(c-a)(e+g), M7=(b-d)(f+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    #
    # U indices: a=0, b=1, c=2, d=3
    # For block k: V rows e=k, f=k+1, g=13+k, h=13+k+1
    # W rows: C00=2k, C01=2k+1, C10=2k+2, C11=2k+3

    strassen_U = [
        [1, 0, 0, 1],   # M1: (a+d)
        [0, 0, 1, 1],   # M2: (c+d)
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: (a+b)
        [0, 0, 1, -1],  # M6: (c-a)
        [0, 1, 0, -1],  # M7: (b-d)
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: (e+h)
        [0, 1, 0, 0],   # M2: f
        [0, 0, 1, -1],  # M3: (g-h)
        [1, -1, 0, 0],  # M4: (e-f)
        [0, 0, 0, 1],   # M5: h
        [1, 0, 1, 0],   # M6: (e+g)
        [0, 1, 0, 1],   # M7: (f+h)
    ]
    # W coefficients [C00, C01, C10, C11] for each Strassen factor:
    strassen_W = [
        [1, 0, 0, 1],   # M1: +C00, +C11
        [0, 0, 1, -1],  # M2: +C10, -C11
        [0, 1, 0, 1],   # M3: +C01, +C11
        [1, 0, 1, 0],   # M4: +C00, +C10
        [-1, 1, 0, 0],  # M5: -C00, +C01
        [0, 0, 0, 1],   # M6: +C11
        [1, 0, 0, 0],   # M7: +C00
    ]

    col = 0
    for k in range(6):
        v_rows = [k, k + 1, 13 + k, 13 + k + 1]
        w_rows = [2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3]
        for r in range(7):
            for u_idx in range(4):
                U[u_idx][col] = strassen_U[r][u_idx]
            for vr_idx in range(4):
                V[v_rows[vr_idx]][col] = strassen_V[r][vr_idx]
            for wr_idx in range(4):
                W[w_rows[wr_idx]][col] = strassen_W[r][wr_idx]
            col += 1

    # Schoolbook for column 12: (a,b;c,d)*(x;y), x=B[0,12], y=B[1,12]
    # P1=ax, P2=by, P3=cx, P4=dy; C[0,12]=P1+P2, C[1,12]=P3+P4
    # V: x at row 12, y at row 25; W: C[0,12] at row 24, C[1,12] at row 25

    schoolbook_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    schoolbook_V = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
    ]
    schoolbook_W = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ]

    for r in range(4):
        for u_idx in range(4):
            U[u_idx][col] = schoolbook_U[r][u_idx]
        V[12][col] = schoolbook_V[r][0]
        V[25][col] = schoolbook_V[r][1]
        W[24][col] = schoolbook_W[r][0]
        W[25][col] = schoolbook_W[r][1]
        col += 1

    return {"U": U, "V": V, "W": W}
