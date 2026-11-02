"""
5x5 matrix multiplication tensor decomposition.
Block decomposition with Strassen 2x2 and Laderman 3x3.
Total rank: 7 + 15 + 12 + 12 + 12 + 18 + 18 + 18 = 112.
"""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    _t0 = time.time()
    DEADLINE = _t0 + 120.0

    n, m, p = DIMENSIONS  # 5, 5, 5

    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    expected = 1.0 if i == k else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    def schoolbook_block(n1, m1, p1, U, V, W, i0, j0, k0, start_col):
        col = start_col
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[(i0+i)*m + (j0+j)][col] = 1
                    V[(j0+j)*p + (k0+k)][col] = 1
                    W[(k0+k)*n + (i0+i)][col] = 1
                    col += 1
        return col

    # Strassen 2x2: 7 terms
    # U index within block: i*2+j; V: j*2+k; W: k*2+i
    strassen_2x2 = [
        ((1,0,0,1),(1,0,0,1),(1,0,0,1)),   # (a+d)(e+h)
        ((0,0,1,1),(1,0,0,0),(0,1,0,-1)),  # (c+d)e
        ((1,0,0,0),(0,1,0,-1),(0,0,1,1)),  # a(f-h)
        ((0,0,0,1),(-1,0,1,0),(1,1,0,0)),  # d(g-e)
        ((1,1,0,0),(0,0,0,1),(-1,0,1,0)),  # (a+b)h
        ((-1,0,1,0),(1,1,0,0),(0,0,0,1)),  # (c-a)(e+f)
        ((0,1,0,-1),(0,0,1,1),(1,0,0,0)),  # (b-d)(g+h)
    ]

    # Laderman 3x3: 15 terms
    # U index within block: i*3+j; V: j*3+k; W: k*3+i
    laderman_3x3 = [
        ((1,1,0,0,0,0,0,0,0),(1,0,0,0,1,0,0,0,0),(1,1,0,0,-1,0,1,0,1)),
        ((1,-1,0,0,0,0,0,0,0),(1,0,0,0,-1,0,0,0,0),(1,-1,0,0,0,0,-1,0,0)),
        ((0,0,1,0,0,1,0,0,0),(0,0,0,0,0,0,1,0,1),(0,0,0,0,0,0,1,1,0)),
        ((0,0,1,0,0,-1,0,0,0),(0,0,0,0,0,0,1,0,-1),(0,0,0,0,0,0,0,-1,1)),
        ((0,0,0,0,1,1,0,0,0),(0,0,0,0,0,0,0,1,1),(0,1,0,0,1,0,0,0,1)),
        ((0,0,0,0,1,-1,0,0,0),(0,0,0,0,0,0,0,1,-1),(0,-1,0,0,0,0,0,0,-1)),
        ((1,0,0,0,1,0,0,0,0),(1,0,0,0,0,1,0,0,0),(1,0,0,1,0,0,0,0,0)),
        ((1,0,0,0,-1,0,0,0,0),(1,0,0,0,0,-1,0,0,0),(-1,0,0,0,1,0,0,0,0)),
        ((0,0,0,0,1,0,0,0,1),(0,0,0,0,0,1,0,0,1),(0,0,1,0,0,0,1,1,0)),
        ((0,0,0,0,1,0,0,0,-1),(0,0,0,0,0,1,0,0,-1),(0,0,-1,0,0,0,-1,-1,0)),
        ((0,1,0,0,0,1,0,0,0),(0,1,0,0,0,1,0,0,0),(0,0,0,1,0,1,0,0,0)),
        ((0,1,0,0,0,-1,0,0,0),(0,1,0,0,0,-1,0,0,0),(0,0,0,-1,1,0,-1,0,0)),
        ((0,0,0,0,0,0,1,1,0),(0,0,0,0,0,0,1,1,0),(0,0,1,0,0,1,0,0,0)),
        ((0,0,0,0,0,0,-1,1,0),(0,0,0,0,0,0,-1,1,0),(0,0,-1,0,0,-1,0,0,0)),
        ((0,1,0,0,0,0,0,1,0),(0,1,0,0,0,0,0,1,0),(0,0,0,-1,1,0,0,0,-1)),
    ]

    total_rank = 7 + 15 + 12 + 12 + 12 + 18 + 18 + 18  # 112
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    col = 0

    # Block 1: A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2]  (Strassen, 7)
    for (u, v, w) in strassen_2x2:
        for idx in range(4):
            U[idx][col] += u[idx]
            V[idx][col] += v[idx]
            W[idx][col] += w[idx]
        col += 1

    # Block 2: A[2:5,2:5]*B[2:5,2:5] -> C[2:5,2:5]  (Laderman, 15)
    for (u, v, w) in laderman_3x3:
        for i in range(3):
            for j in range(3):
                U[(2+i)*5+(2+j)][col] += u[i*3+j]
        for j in range(3):
            for k in range(3):
                V[(2+j)*5+(2+k)][col] += v[j*3+k]
        for k in range(3):
            for i in range(3):
                W[(2+k)*5+(2+i)][col] += w[k*3+i]
        col += 1

    # Off-diagonal blocks (schoolbook)
    col = schoolbook_block(2, 3, 2, U, V, W, 0, 2, 0, col)   # A[0:2,2:5]*B[2:5,0:2]->C[0:2,0:2]  (12)
    col = schoolbook_block(2, 2, 3, U, V, W, 0, 0, 2, col)   # A[0:2,0:2]*B[0:2,2:5]->C[0:2,2:5]  (12)
    col = schoolbook_block(3, 2, 2, U, V, W, 2, 0, 0, col)   # A[2:5,0:2]*B[0:2,0:2]->C[2:5,0:2]  (12)
    col = schoolbook_block(3, 3, 2, U, V, W, 2, 2, 0, col)   # A[2:5,2:5]*B[2:5,0:2]->C[2:5,0:2]  (18)
    col = schoolbook_block(2, 3, 3, U, V, W, 0, 2, 2, col)   # A[0:2,2:5]*B[2:5,2:5]->C[0:2,2:5]  (18)
    col = schoolbook_block(3, 2, 3, U, V, W, 2, 0, 2, col)   # A[2:5,0:2]*B[0:2,2:5]->C[2:5,2:5]  (18)

    # Verify
    if not verify(U, V, W, n, m, p):
        # Fallback: pure schoolbook 125
        rk = n * m * p
        U = [[0]*rk for _ in range(n*m)]
        V = [[0]*rk for _ in range(m*p)]
        W = [[0]*rk for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
