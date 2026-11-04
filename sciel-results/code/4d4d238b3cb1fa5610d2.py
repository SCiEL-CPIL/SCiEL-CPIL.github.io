DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # U: (n*m, rank) index i*m+j ; V: (m*p, rank) index j*p+k ; W: (p*n, rank) index k*n+i
    U_cols = []  # each col is a dict idx->coeff
    V_cols = []
    W_cols = []
    
    def add(u, v, w):
        # u, v, w are dicts idx->coeff
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Strassen M1..M7: (a[i'][j'], b[j'][k'], c[i'][k'])
    S = [
        ([[0,-1],[0,1]], [[1,1],[0,0]], [[1,0],[0,1]]),
        ([[1,1],[0,0]],  [[1,0],[0,0]], [[1,1],[0,0]]),
        ([[0,0],[1,1]],  [[0,0],[0,1]], [[0,1],[0,-1]]),
        ([[1,0],[-1,0]], [[0,0],[1,1]], [[-1,0],[1,0]]),
        ([[1,0],[0,1]],  [[1,0],[0,-1]], [[1,0],[0,1]]),
        ([[-1,0],[1,0]], [[0,1],[0,1]], [[0,1],[0,0]]),
        ([[0,-1],[0,1]], [[-1,0],[1,0]], [[0,0],[0,1]]),
    ]
    
    def Uidx(i, j): return i * m + j
    def Vidx(j, k): return j * p + k
    def Widx(k, i): return k * n + i
    
    def block_1x2x2(ai, aj0, bj0, bk0):
        # A block 1x2 at row ai, cols aj0, aj0+1
        # B block 2x2 at rows bj0,bj0+1, cols bk0,bk0+1
        # C block 1x2 at row ai, cols bk0,bk0+1
        add({Uidx(ai, aj0): 1}, {Vidx(bj0, bk0): 1}, {Widx(bk0, ai): 1})
        add({Uidx(ai, aj0): 1}, {Vidx(bj0, bk0+1): 1}, {Widx(bk0+1, ai): 1})
        add({Uidx(ai, aj0+1): 1}, {Vidx(bj0+1, bk0): 1}, {Widx(bk0, ai): 1})
        add({Uidx(ai, aj0+1): 1}, {Vidx(bj0+1, bk0+1): 1}, {Widx(bk0+1, ai): 1})
    
    def block_2x2x2(ai0, aj0, bj0, bk0):
        # A block 2x2 at rows ai0,ai0+1, cols aj0,aj0+1
        # B block 2x2 at rows bj0,bj0+1, cols bk0,bk0+1
        # C block 2x2 at rows ai0,ai0+1, cols bk0,bk0+1
        for a, b, c in S:
            u, v, w = {}, {}, {}
            for i2 in range(2):
                for j2 in range(2):
                    if a[i2][j2] != 0:
                        idx = Uidx(ai0+i2, aj0+j2)
                        u[idx] = u.get(idx, 0) + a[i2][j2]
            for j2 in range(2):
                for k2 in range(2):
                    if b[j2][k2] != 0:
                        idx = Vidx(bj0+j2, bk0+k2)
                        v[idx] = v.get(idx, 0) + b[j2][k2]
            for i2 in range(2):
                for k2 in range(2):
                    if c[i2][k2] != 0:
                        idx = Widx(bk0+k2, ai0+i2)
                        w[idx] = w.get(idx, 0) + c[i2][k2]
            add(u, v, w)
    
    # For each column group g (cols 2g, 2g+1 of B and C)
    for g in range(3):
        bk0 = 2 * g  # start col of B and C for this group
        # C_top: row ai=0
        #   a1 = A[0, 0:2], B_g_top = B[0:2, bk0:bk0+2]
        block_1x2x2(ai=0, aj0=0, bj0=0, bk0=bk0)
        #   a2 = A[0, 2:4], B_g_bottom = B[2:4, bk0:bk0+2]
        block_1x2x2(ai=0, aj0=2, bj0=2, bk0=bk0)
        # C_bottom: rows ai0=1,2
        #   A1 = A[1:3, 0:2], B_g_top = B[0:2, bk0:bk0+2]
        block_2x2x2(ai0=1, aj0=0, bj0=0, bk0=bk0)
        #   A2 = A[1:3, 2:4], B_g_bottom = B[2:4, bk0:bk0+2]
        block_2x2x2(ai0=1, aj0=2, bj0=2, bk0=bk0)
    
    rank = len(U_cols)
    # Build U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx][c] = val
        for idx, val in V_cols[c].items():
            V[idx][c] = val
        for idx, val in W_cols[c].items():
            W[idx][c] = val
    return {"U": U, "V": V, "W": W}
