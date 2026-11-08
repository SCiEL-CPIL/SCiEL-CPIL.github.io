import time

DIMENSIONS = [4, 5, 6]

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 Strassen, each shape (4,7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a00+a11)*(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a10+a11)*b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a00*(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a11*(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a10-a00)*(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a01-a11)*(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """4x4x4 Strassen, rank 49. U:(16,49), V:(16,49), W:(16,49)."""
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    strassen_defs = [
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            for li in range(2):
                for lj in range(2):
                    la = li * 2 + lj
                    ca = U2[la][t]
                    if ca:
                        for (bi, bj), sig in a_blocks:
                            gi = bi * 2 + li
                            gj = bj * 2 + lj
                            U[gi * 4 + gj][col] += ca * sig
            for lj in range(2):
                for lk in range(2):
                    lb = lj * 2 + lk
                    cb = V2[lb][t]
                    if cb:
                        for (bj, bk), sig in b_blocks:
                            gj = bj * 2 + lj
                            gk = bk * 2 + lk
                            V[gj * 4 + gk][col] += cb * sig
            for lk in range(2):
                for li in range(2):
                    lc = lk * 2 + li
                    cc = W2[lc][t]
                    if cc:
                        for (bi, bk), sig in c_blocks:
                            gk = bk * 2 + lk
                            gi = bi * 2 + li
                            W[gk * 4 + gi][col] += cc * sig
    return U, V, W

def _build_4x4x2_strassen():
    """4x4x2 using 4 Strassen 2x2x2 blocks. Rank 28. U:(16,28), V:(8,28), W:(8,28)."""
    U2, V2, W2 = _strassen_2x2()
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    def add_block(ai, aj, bj, ci, ck, base):
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    ca = U2[li*2+lj][t]
                    if ca:
                        U[(ai+li)*4 + (aj+lj)][col] += ca
            for lj in range(2):
                for lk in range(2):
                    cb = V2[lj*2+lk][t]
                    if cb:
                        V[(bj+lj)*2 + (ck+lk)][col] += cb
            for lk in range(2):
                for li in range(2):
                    cc = W2[lk*2+li][t]
                    if cc:
                        W[(ck+lk)*4 + (ci+li)][col] += cc
    
    add_block(0, 0, 0, 0, 0, 0)
    add_block(0, 2, 2, 0, 0, 7)
    add_block(2, 0, 0, 2, 0, 14)
    add_block(2, 2, 2, 2, 0, 21)
    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine(decomps, n, m, p):
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    cur = 0
    for Us, Vs, Ws, (is_, ie), (js_, je), (ks_, ke) in decomps:
        ns, ms, ps = ie-is_, je-js_, ke-ks_
        rs = len(Us[0])
        for r in range(ns*ms):
            il, jl = divmod(r, ms)
            U[(is_+il)*m + (js_+jl)][cur:cur+rs] = Us[r]
        for r in range(ms*ps):
            jl, kl = divmod(r, ps)
            V[(js_+jl)*p + (ks_+kl)][cur:cur+rs] = Vs[r]
        for r in range(ps*ns):
            kl, il = divmod(r, ns)
            W[(ks_+kl)*n + (is_+il)][cur:cur+rs] = Ws[r]
        cur += rs
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 110.0
    n, m, p = DIMENSIONS
    
    try:
        U1, V1, W1 = _build_4x4x4_strassen()
        d1 = (U1, V1, W1, (0,4), (0,4), (0,4))
        
        U2, V2, W2 = _build_schoolbook(4, 1, 6)
        d2 = (U2, V2, W2, (0,4), (4,5), (0,6))
        
        U3, V3, W3 = _build_4x4x2_strassen()
        d3 = (U3, V3, W3, (0,4), (0,4), (4,6))
        
        U, V, W = _combine([d1, d2, d3], n, m, p)
        return {"U": U, "V": V, "W": W}
    except:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
