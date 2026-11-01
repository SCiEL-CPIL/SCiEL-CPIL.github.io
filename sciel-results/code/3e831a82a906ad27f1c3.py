import time

DIMENSIONS = [2, 2, 12]

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    R = len(U[0]) if len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - (1.0 if j == i else 0.0)) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_t = time.time()
    deadline = start_t + 105  # hard safety margin before the 150s kill
    n, m, p = DIMENSIONS  # 2, 2, 12

    # Fallback is always a valid rank-48 schoolbook decomposition.
    fb = _schoolbook(n, m, p)

    # Rank-42 decomposition: split the 12 columns into 6 pairs of 2 and apply
    # Strassen's 7-product scheme to each 2x2x2 block.
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42

    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0.0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]   # 24 x 42

    # Strassen 7 products for A=[[a,b],[c,d]] (A[i][j]) * B=[[e,f],[g,h]] (B[j][k]):
    #   M1=(a+d)(e+h)   M2=(c+d)e   M3=a(f-h)   M4=d(g-e)
    #   M5=(a+b)h       M6=(c-a)(e+f)  M7=(b-d)(g+h)
    #   C00=M1+M4-M5+M7   C01=M3+M5   C10=M2+M4   C11=M1-M2+M3+M6
    # For each pair t: columns k0=2t (e,f) and k1=2t+1 (g,h).
    # A-index rows: a=A[0][0]->U0, b=A[0][1]->U1, c=A[1][0]->U2, d=A[1][1]->U3
    # B-index rows: e=B[0][k0]->V[k0], f=B[0][k1]->V[k1], g=B[1][k0]->V[p+k0], h=B[1][k1]->V[p+k1]
    # C-index rows: C00->W[2k0], C01->W[2k0+1], C10->W[2k1], C11->W[2k1+1]  (k*n+i, n=2)

    for t in range(num_pairs):
        base = t * 7
        k0 = 2 * t
        k1 = 2 * t + 1
        e = k0          # B[0][k0]
        f = k1          # B[0][k1]
        g = p + k0      # B[1][k0]
        h = p + k1      # B[1][k1]
        C00 = 2 * k0    # W index for C[0][k0]
        C01 = 2 * k0 + 1
        C10 = 2 * k1
        C11 = 2 * k1 + 1

        def setU(row, val, s):
            U[row][base + s] = float(val)
        def setV(row, val, s):
            V[row][base + s] = float(val)
        def setW(row, val, s):
            W[row][base + s] = float(val)

        # M1=(a+d)(e+h)
        setU(0, 1, 0); setU(3, 1, 0)
        setV(e, 1, 0); setV(h, 1, 0)
        setW(C00, 1, 0); setW(C11, 1, 0)
        # M2=(c+d)e
        setU(2, 1, 1); setU(3, 1, 1)
        setV(e, 1, 1)
        setW(C10, 1, 1); setW(C11, -1, 1)
        # M3=a(f-h)
        setU(0, 1, 2)
        setV(f, 1, 2); setV(h, -1, 2)
        setW(C01, 1, 2); setW(C11, 1, 2)
        # M4=d(g-e)
        setU(3, 1, 3)
        setV(g, 1, 3); setV(e, -1, 3)
        setW(C00, 1, 3); setW(C10, 1, 3)
        # M5=(a+b)h
        setU(0, 1, 4); setU(1, 1, 4)
        setV(h, 1, 4)
        setW(C00, -1, 4); setW(C01, 1, 4)
        # M6=(c-a)(e+f)
        setU(2, 1, 5); setU(0, -1, 5)
        setV(e, 1, 5); setV(f, 1, 5)
        setW(C11, 1, 5)
        # M7=(b-d)(g+h)
        setU(1, 1, 6); setU(3, -1, 6)
        setV(g, 1, 6); setV(h, 1, 6)
        setW(C00, 1, 6)

    # Verify the rank-42 candidate; fall back to schoolbook if it fails or we
    # are near the deadline.
    if time.time() < deadline and _check(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    return fb
# EVOLVE-BLOCK-END
