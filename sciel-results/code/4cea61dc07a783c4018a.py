import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Fallback: rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _strassen_2x2():
    """(2x2)*(2x2) rank 7."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    defs = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (0, 1, -1), (1, 0, -1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(1, 0, 1), (0, 0, -1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]
    for t, (ue, ve, we) in enumerate(defs):
        for (i, j, c) in ue:
            U[i * 2 + j][t] += c
        for (j, k, c) in ve:
            V[j * 2 + k][t] += c
        for (k, i, c) in we:
            W[k * 2 + i][t] += c
    return U, V, W


def _build_4x4x4():
    """(4x4)*(4x4) rank 49 via 2-level Strassen."""
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    U2, V2, W2 = _strassen_2x2()
    outer = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(1, 0, 1), (0, 0, -1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]
    for s, (ue, ve, we) in enumerate(outer):
        for t in range(7):
            col = s * 7 + t
            for (bi, bj, c) in ue:
                for i in range(2):
                    for j in range(2):
                        gi = bi * 2 + i
                        gj = bj * 2 + j
                        a_loc = i * 2 + j
                        cc = U2[a_loc][t]
                        if cc:
                            U[gi * 4 + gj][col] += cc * c
            for (bj, bk, c) in ve:
                for j in range(2):
                    for k in range(2):
                        gj = bj * 2 + j
                        gk = bk * 2 + k
                        b_loc = j * 2 + k
                        cc = V2[b_loc][t]
                        if cc:
                            V[gj * 4 + gk][col] += cc * c
            for (bk, bi, c) in we:
                for k in range(2):
                    for i in range(2):
                        gk = bk * 2 + k
                        gi = bi * 2 + i
                        c_loc = k * 2 + i
                        cc = W2[c_loc][t]
                        if cc:
                            W[gk * 4 + gi][col] += cc * c
    return U, V, W


def _build_4x4x2():
    """(4x4)*(4x2) rank 28 via 4 blocks of 2x2x2 Strassen."""
    U = [[0] * 28 for _ in range(16)]
    V = [[0] * 28 for _ in range(8)]
    W = [[0] * 28 for _ in range(8)]
    U2, V2, W2 = _strassen_2x2()
    blocks = [(0, 0, 0, 0, 0), (0, 2, 2, 0, 0), (2, 0, 0, 2, 0), (2, 2, 2, 2, 0)]
    for bi, (ar, ac, br, cr, ck) in enumerate(blocks):
        base = bi * 7
        for t in range(7):
            col = base + t
            for i in range(2):
                for j in range(2):
                    a_loc = i * 2 + j
                    cc = U2[a_loc][t]
                    if cc:
                        U[(ar + i) * 4 + (ac + j)][col] += cc
            for j in range(2):
                for k in range(2):
                    b_loc = j * 2 + k
                    cc = V2[b_loc][t]
                    if cc:
                        V[(br + j) * 2 + (ck + k)][col] += cc
            for k in range(2):
                for i in range(2):
                    c_loc = k * 2 + i
                    cc = W2[c_loc][t]
                    if cc:
                        W[(ck + k) * 4 + (cr + i)][col] += cc
    return U, V, W


def _build_4x4x1():
    """(4x4)*(4x1) rank 16 (schoolbook, no Strassen saving)."""
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _build_4x4x3():
    """(4x4)*(4x3) rank 44 = 28 (4x4x2) + 16 (4x4x1)."""
    U2, V2, W2 = _build_4x4x2()
    U1, V1, W1 = _build_4x4x1()
    rank = len(U2[0]) + len(U1[0])
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    for i in range(16):
        for c in range(len(U2[0])):
            U[i][c] = U2[i][c]
    for i in range(8):
        for c in range(len(V2[0])):
            V[i][c] = V2[i][c]
    for i in range(8):
        for c in range(len(W2[0])):
            W[i][c] = W2[i][c]
    off = len(U2[0])
    for i in range(16):
        for c in range(len(U1[0])):
            U[i][off + c] = U1[i][c]
    for i in range(4):
        for c in range(len(V1[0])):
            V[8 + i][off + c] = V1[i][c]
    for i in range(4):
        for c in range(len(W1[0])):
            W[8 + i][off + c] = W1[i][c]
    return U, V, W


def _build_4x2x6():
    """(4x2)*(2x6) rank 44 = 2*(4x2x3). 4x2x3 = 14 (4x2x2) + 8 (4x2x1)."""
    # 4x2x2: A is 4x2, B is 2x2. Use 2 blocks of 2x2x2 Strassen.
    # 4x2x1: A is 4x2, B is 2x1. Schoolbook: 8.
    def _build_4x2x2():
        U = [[0] * 14 for _ in range(8)]
        V = [[0] * 14 for _ in range(4)]
        W = [[0] * 14 for _ in range(8)]
        U2, V2, W2 = _strassen_2x2()
        for bi in range(2):
            base = bi * 7
            for t in range(7):
                col = base + t
                for i in range(2):
                    for j in range(2):
                        a_loc = i * 2 + j
                        cc = U2[a_loc][t]
                        if cc:
                            U[(bi * 2 + i) * 2 + j][col] += cc
                for j in range(2):
                    for k in range(2):
                        b_loc = j * 2 + k
                        cc = V2[b_loc][t]
                        if cc:
                            V[j * 2 + k][col] += cc
                for k in range(2):
                    for i in range(2):
                        c_loc = k * 2 + i
                        cc = W2[c_loc][t]
                        if cc:
                            W[k * 2 + (bi * 2 + i)][col] += cc
        return U, V, W

    def _build_4x2x1():
        n, m, p = 4, 2, 1
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return U, V, W

    U2, V2, W2 = _build_4x2x2()
    U1, V1, W1 = _build_4x2x1()
    rank = len(U2[0]) + len(U1[0])
    U = [[0] * (2 * rank) for _ in range(8)]
    V = [[0] * (2 * rank) for _ in range(12)]
    W = [[0] * (2 * rank) for _ in range(24)]
    for part in range(2):
        off = part * rank
        for i in range(8):
            for c in range(len(U2[0])):
                U[i][off + c] = U2[i][c]
        for i in range(4):
            for c in range(len(V2[0])):
                V[i + part * 4][off + c] = V2[i][c]
        for i in range(8):
            for c in range(len(W2[0])):
                W[i + part * 8][off + c] = W2[i][c]
        for i in range(8):
            for c in range(len(U1[0])):
                U[i][off + len(U2[0]) + c] = U1[i][c]
        for i in range(2):
            for c in range(len(V1[0])):
                V[4 + part * 4 + i][off + len(U2[0]) + c] = V1[i][c]
        for i in range(4):
            for c in range(len(W1[0])):
                W[8 + part * 8 + i][off + len(U2[0]) + c] = W1[i][c]
    return U, V, W


def _build_4x5x6_strassen():
    """
    (4x5)*(5x6) using Strassen-based decomposition.
    
    Pad A to 4x6 (add zero col), B to 6x6 (add zero row).
    A' = [A11 A12] where A11 is 4x4, A12 is 4x2
    B' = [B11; B12] where B11 is 4x6, B12 is 2x6
    
    C = A11*B11 + A12*B12
    
    A11*B11: 4x4 * 4x6 = (4x4*4x3) + (4x4*4x3) = 44 + 44 = 88? No.
    Actually B11 is 4x6, so split into two 4x3 blocks.
    A11*B11 = A11*B11a + A11*B11b where B11a, B11b are 4x3.
    Each 4x4*4x3 has rank 44. Total: 88.
    
    A12*B12: 4x2 * 2x6. Rank 44.
    
    Total: 88 + 44 = 132. That's worse than schoolbook!
    
    Let me reconsider. Maybe I should use a different split.
    
    Alternative: A (4x5) = [A1 A2] where A1 is 4x2, A2 is 4x3
    B (5x6) = [B1; B2] where B1 is 2x6, B2 is 3x6
    
    C = A1*B1 + A2*B2
    
    A1*B1: 4x2 * 2x6 = rank 44
    A2*B2: 4x3 * 3x6. Split A2 into [A21; A22] (2x3 each), B2 into [B21 B22] (3x3 each).
    A2*B2 = A21*B21 + A21*B22 + A22*B21 + A22*B22
    Each 2x3*3x3: rank 18. Total: 4*18 = 72.
    
    Total: 44 + 72 = 116. Still worse than 120? No, 116 < 120.
    
    Wait, let me recalculate. 44 + 72 = 116. That's better than 120!
    """
    n, m, p = 4, 5, 6
    
    parts = []
    
    # A1*B1: 4x2 * 2x6, rank 44
    U1, V1, W1 = _build_4x2x6()
    parts.append((U1, V1, W1, (0, 0, 4, 2, 0, 6)))
    
    # A2*B2: 4x3 * 3x6
    # Split A2 into [A21; A22] (2x3 each), B2 into [B21 B22] (3x3 each)
    # A2*B2 = A21*B21 + A21*B22 + A22*B21 + A22*B22
    for i_block in range(2):
        for k_block in range(2):
            U_s, V_s, W_s = _build_schoolbook(2, 3, 3)
            i0 = i_block * 2
            k0 = k_block * 3
            parts.append((U_s, V_s, W_s, (i0, 2, 2, 3, k0, 3)))
    
    U, V, W = _combine_parts(parts, n, m, p)
    return U, V, W


def _combine_parts(parts, n, m, p):
    """Combine (U,V,W, (i0,j0,ii,jj,k0,kk)) parts into full (n,m,p) decomposition."""
    total = sum(len(d[0][0]) for d in parts)
    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]
    off = 0
    for (Us, Vs, Ws, (i0, j0, ii, jj, k0, kk)) in parts:
        nA = ii
        mA = jj
        nB = kk
        for r in range(nA * mA):
            i = r // mA
            j = r % mA
            for c in range(len(Us[0])):
                U[(i0 + i) * m + (j0 + j)][off + c] = Us[r][c]
        for r in range(mA * nB):
            j = r // nB
            k = r % nB
            for c in range(len(Vs[0])):
                V[(j0 + j) * p + (k0 + k)][off + c] = Vs[r][c]
        for r in range(nB * nA):
            k = r // nA
            i = r % nA
            for c in range(len(Ws[0])):
                W[(k0 + k) * n + (i0 + i)][off + c] = Ws[r][c]
        off += len(Us[0])
    return U, V, W


def _verify(U, V, W, n, m, p, trials=5, seed=12345):
    import random
    rng = random.Random(seed)
    rank = len(U[0])
    for _ in range(trials):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                val = 0
                for t in range(rank):
                    au = sum(U[i * m + j][t] * A[i][j] for j in range(m))
                    bu = sum(V[j * p + k][t] * B[j][k] for j in range(m))
                    val += W[k * n + i][t] * au * bu
                if val != C[i][k]:
                    return False
    return True


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 120

    try:
        # Try the Strassen-based decomposition
        U, V, W = _build_4x5x6_strassen()
        rank = len(U[0])
        
        if time.time() > deadline - 20:
            U, V, W = _build_schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
        
        if _verify(U, V, W, n, m, p, trials=5):
            if time.time() > deadline - 10:
                return {"U": U, "V": V, "W": W}
            school_rank = n * m * p
            if rank < school_rank:
                return {"U": U, "V": V, "W": W}
        
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    except Exception:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
