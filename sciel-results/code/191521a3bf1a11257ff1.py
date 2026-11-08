import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """(2x2)*(2x2) rank 7. U/V/W are 4x7 (row = i*2+j / j*2+k / k*2+i)."""
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


def _build_2x3x3():
    """(2x3)*(3x3) rank 18 schoolbook."""
    n, m, p = 2, 3, 3
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


def _build_2x2x3():
    """(2x2)*(2x3) rank 12 schoolbook."""
    n, m, p = 2, 2, 3
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
    """(4x4)*(4x1) rank 16 (schoolbook)."""
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
                W[8 + part * 8 + i][off + len(W1[0]) + c] = W1[i][c]
    return U, V, W


def _build_4x3x6():
    """
    (4x3)*(3x6) rank 72 via 4 blocks of 2x3x3 schoolbook.
    A = [[A11, A12], [A21, A22]] where A11,A21 are 2x3 (rows split)
    Wait, A is 4x3, B is 3x6.
    Split rows of A: A = [A_top (2x3); A_bot (2x3)]
    Split cols of B: B = [B_left (3x3) B_right (3x3)]
    C = A_top * B_left + A_top * B_right + A_bot * B_left + A_bot * B_right
    Each term is (2x3)*(3x3) with rank 18.
    Total: 4 * 18 = 72.
    """
    parts = []
    for i_block in range(2):
        for k_block in range(2):
            U_s, V_s, W_s = _build_2x3x3()
            i0 = i_block * 2
            k0 = k_block * 3
            parts.append((U_s, V_s, W_s, (i0, 0, 2, 3, k0, 3)))
    return _combine_parts(parts, 4, 3, 6)


def _build_2x5x6():
    """
    (2x5)*(5x6) rank 60 via schoolbook.
    Actually, let's try to do better.
    Split cols of A: A = [A1 (2x2) A2 (2x3)]
    Split rows of B: B = [B1 (2x6); B2 (3x6)]
    C = A1*B1 + A2*B2
    A1*B1: (2x2)*(2x6). Rank?
    A2*B2: (2x3)*(3x6). Rank?
    
    (2x2)*(2x6): Split B1 into [B1a (2x3) B1b (2x3)].
    (2x2)*(2x3) rank 12. Two of them: 24.
    (2x3)*(3x6): Split B2 into [B2a (3x3) B2b (3x3)].
    (2x3)*(3x3) rank 18. Two of them: 36.
    Total: 24 + 36 = 60. Same as schoolbook.
    
    Let's just use schoolbook for simplicity and correctness.
    """
    n, m, p = 2, 5, 6
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


def _build_4x5x6_kronecker():
    """
    Kronecker-style decomposition of (4x5)*(5x6).
    
    Structural idea: Treat 4x5x6 as a product of smaller tensor factors.
    We can view the 4x5 matrix as a 2x2 block matrix where each block is 2x(2.5)? No.
    
    Alternative structural approach:
    Split the 4x5 matrix A into two parts along columns:
    A = [A_left (4x2) | A_right (4x3)]
    Split the 5x6 matrix B into two parts along rows:
    B = [B_top (2x6); B_bottom (3x6)]
    
    Then C = A_left * B_top + A_right * B_bottom.
    
    Part 1: (4x2)*(2x6) -> rank 44 (from _build_4x2x6)
    Part 2: (4x3)*(3x6) -> rank 72 (from _build_4x3x6)
    
    Total rank: 44 + 72 = 116.
    
    But wait, we can do better on Part 2.
    (4x3)*(3x6): 
    Split A_right into [A_r1 (2x3); A_r2 (2x3)]
    Split B_bottom into [B_b1 (3x3) B_b2 (3x3)]
    C2 = A_r1*B_b1 + A_r1*B_b2 + A_r2*B_b1 + A_r2*B_b2
    Each (2x3)*(3x3) has rank 18. Total: 72.
    
    Can we do better on Part 1?
    (4x2)*(2x6):
    Split A_left into [A_l1 (2x2); A_l2 (2x2)]
    Split B_top into [B_t1 (2x3) B_t2 (2x3)]
    C1 = A_l1*B_t1 + A_l1*B_t2 + A_l2*B_t1 + A_l2*B_t2
    Each (2x2)*(2x3) has rank 12. Total: 48.
    But _build_4x2x6 gives 44. Let me check why.
    
    _build_4x2x6: 2*(4x2x3). 4x2x3 = 14 (4x2x2) + 8 (4x2x1).
    4x2x2: 2 blocks of 2x2x2 Strassen = 14.
    4x2x1: schoolbook = 8.
    Total per part: 22. Two parts: 44.
    
    My alternative: 4 * 12 = 48. Worse.
    So _build_4x2x6 is better.
    
    Total: 44 + 72 = 116.
    """
    parts = []
    
    # Part 1: (4x2)*(2x6) at position (i=0, j=0) to (i=4, j=2) and (j=0, k=0) to (j=2, k=6)
    U1, V1, W1 = _build_4x2x6()
    parts.append((U1, V1, W1, (0, 0, 4, 2, 0, 6)))
    
    # Part 2: (4x3)*(3x6) at position (i=0, j=2) to (i=4, j=5) and (j=2, k=0) to (j=5, k=6)
    U2, V2, W2 = _build_4x3x6()
    parts.append((U2, V2, W2, (0, 2, 4, 3, 2, 6)))
    
    return _combine_parts(parts, 4, 5, 6)


def _build_4x5x6_additive():
    """
    Additive Strassen-based decomposition.
    A = [A11 (4x4) A12 (4x1)]
    B = [B11 (4x6); B12 (1x6)]
    C = A11*B11 + A12*B12
    
    A11*B11: (4x4)*(4x6) = (4x4)*(4x3) + (4x4)*(4x3) = 44 + 44 = 88
    A12*B12: (4x1)*(1x6) = rank 24 (schoolbook)
    
    Total: 88 + 24 = 112.
    """
    parts = []
    
    # A11*B11a: (4x4)*(4x3)
    U1, V1, W1 = _build_4x4x3()
    parts.append((U1, V1, W1, (0, 0, 4, 4, 0, 3)))
    
    # A11*B11b: (4x4)*(4x3)
    U2, V2, W2 = _build_4x4x3()
    parts.append((U2, V2, W2, (0, 0, 4, 4, 3, 3)))
    
    # A12*B12: (4x1)*(1x6)
    n, m, p = 4, 1, 6
    rank = n * m * p
    U3 = [[0] * rank for _ in range(n * m)]
    V3 = [[0] * rank for _ in range(m * p)]
    W3 = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U3[i * m + j][c] = 1
                V3[j * p + k][c] = 1
                W3[k * n + i][c] = 1
                c += 1
    parts.append((U3, V3, W3, (0, 4, 4, 1, 4, 6)))
    
    return _combine_parts(parts, 4, 5, 6)


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


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 110

    candidates = []

    # Try Kronecker-style split: rank 116
    if time.time() < deadline - 30:
        try:
            U, V, W = _build_4x5x6_kronecker()
            rank = len(U[0])
            candidates.append((rank, U, V, W, "kronecker"))
        except Exception:
            pass

    # Try additive Strassen: rank 112
    if time.time() < deadline - 20:
        try:
            U, V, W = _build_4x5x6_additive()
            rank = len(U[0])
            candidates.append((rank, U, V, W, "additive"))
        except Exception:
            pass

    # Schoolbook fallback: rank 120
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    candidates.append((n * m * p, U_sb, V_sb, W_sb, "schoolbook"))

    # Pick the best candidate (lowest rank) that passes verification
    best = None
    for rank, U, V, W, name in candidates:
        if time.time() > deadline - 10:
            break
        if _verify(U, V, W, n, m, p, trials=3):
            if best is None or rank < best[0]:
                best = (rank, U, V, W, name)

    if best is not None:
        return {"U": best[1], "V": best[2], "W": best[3]}

    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
