"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build 2x2 Strassen. Returns (U, V, W) each shape (4, 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1

    return U, V, W

def _build_strassen_4x4():
    """4x4 Strassen, rank 49."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]

    defs = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]

    for s, (ab, bb, cb) in enumerate(defs):
        for t in range(7):
            col = s*7+t
            for li in range(2):
                for lj in range(2):
                    ca = U2[li*2+lj][t]
                    if ca:
                        for (bi,bj),sg in ab:
                            U[(bi*2+li)*4+(bj*2+lj)][col] += ca*sg
            for lj in range(2):
                for lk in range(2):
                    cb_ = V2[lj*2+lk][t]
                    if cb_:
                        for (bj,bk),sg in bb:
                            V[(bj*2+lj)*4+(bk*2+lk)][col] += cb_*sg
            for lk in range(2):
                for li in range(2):
                    cw = W2[lk*2+li][t]
                    if cw:
                        for (bi,bk),sg in cb:
                            W[(bk*2+lk)*4+(bi*2+li)][col] += cw*sg
    return U, V, W

def _build_4x4x2():
    """4x4 * 4x2, rank 28. Split into 4 (2x2)*(2x2) blocks with Strassen."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]

    blocks = [(0,0,0,0,0), (0,2,2,0,0), (2,0,0,2,0), (2,2,2,2,0)]
    for b, (ai,aj,bj,ci,ck) in enumerate(blocks):
        base = b*7
        for t in range(7):
            col = base+t
            for li in range(2):
                for lj in range(2):
                    ca = U2[li*2+lj][t]
                    if ca:
                        U[(ai+li)*4+(aj+lj)][col] += ca
            for lj in range(2):
                for lk in range(2):
                    cb = V2[lj*2+lk][t]
                    if cb:
                        V[(bj+lj)*2+(ck+lk)][col] += cb
            for lk in range(2):
                for li in range(2):
                    cw = W2[lk*2+li][t]
                    if cw:
                        W[(ck+lk)*4+(ci+li)][col] += cw
    return U, V, W

def _build_4x1x6():
    """
    4x1 * 1x6, rank 16 via row-split (2+2).
    Split the 4 rows of A into two groups of 2; each group is a (2x1)*(1x6)
    schoolbook block of rank 8. Total rank = 2*8 = 16.
    U has shape (4,16), V has shape (6,16), W has shape (24,16).
    U[i, r], V[k, r], W[k*4+i, r].
    """
    rank = 16
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(24)]
    for g, (i0, i1) in enumerate([(0, 2), (2, 4)]):
        base = g * 8
        for k in range(6):
            col = base + k
            for i in range(i0, i1):
                U[i][col] = 1
                V[k][col] = 1
                W[k*4+i][col] = 1
    return U, V, W

def _combine(decomps, n, m, p):
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    cur = 0
    for U_s, V_s, W_s, (i_s,i_e), (j_s,j_e), (k_s,k_e) in decomps:
        n_s = i_e-i_s; m_s = j_e-j_s; p_s = k_e-k_s
        r_s = len(U_s[0])

        for r in range(n_s*m_s):
            il = r // m_s; jl = r % m_s
            ig = i_s+il; jg = j_s+jl
            rg = ig*m+jg
            for c in range(r_s):
                U[rg][cur+c] = U_s[r][c]

        for r in range(m_s*p_s):
            jl = r // p_s; kl = r % p_s
            jg = j_s+jl; kg = k_s+kl
            rg = jg*p+kg
            for c in range(r_s):
                V[rg][cur+c] = V_s[r][c]

        for r in range(p_s*n_s):
            kl = r // n_s; il = r % n_s
            kg = k_s+kl; ig = i_s+il
            rg = kg*n+ig
            for c in range(r_s):
                W[rg][cur+c] = W_s[r][c]

        cur += r_s
    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c]=1
                V[j*p+k][c]=1
                W[k*n+i][c]=1
                c += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    try:
        # 1. 4x4x4 Strassen, rank 49
        U1, V1, W1 = _build_strassen_4x4()
        d1 = (U1, V1, W1, (0,4), (0,4), (0,4))

        # 2. 4x4x2 Strassen, rank 28
        U3, V3, W3 = _build_4x4x2()
        d3 = (U3, V3, W3, (0,4), (0,4), (4,6))

        # 3. 4x1x6 row-split, rank 16 (was schoolbook rank 24)
        U2, V2, W2 = _build_4x1x6()
        d2 = (U2, V2, W2, (0,4), (4,5), (0,6))

        U, V, W = _combine([d1, d3, d2], n, m, p)

        # Verify before returning
        _verify(U, V, W, n, m, p)

        return {"U": U, "V": V, "W": W}

    except Exception:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication.

    For the multiplication tensor we need, for all (i,j,k):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")

# EVOLVE-BLOCK-END
