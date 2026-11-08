import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2, rank 7.
    U shape (4,7): row i*2+j, col t
    V shape (4,7): row j*2+k, col t
    W shape (4,7): row k*2+i, col t
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    defs = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1),(1,0,-1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for t, (ue, ve, we) in enumerate(defs):
        for (i, j, c) in ue:
            U[i*2+j][t] += c
        for (j, k, c) in ve:
            V[j*2+k][t] += c
        for (k, i, c) in we:
            W[k*2+i][t] += c
    return U, V, W


def _build_4x4x4():
    """Two-level Strassen for 4x4 * 4x4 -> 4x4, rank 49."""
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    U2, V2, W2 = _strassen_2x2()
    outer = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for s, (ue, ve, we) in enumerate(outer):
        for t in range(7):
            col = s*7 + t
            for (bi, bj, c) in ue:
                for i in range(2):
                    for j in range(2):
                        gi = bi*2 + i
                        gj = bj*2 + j
                        a_loc = i*2 + j
                        cc = U2[a_loc][t]
                        if cc:
                            U[gi*4 + gj][col] += cc * c
            for (bj, bk, c) in ve:
                for j in range(2):
                    for k in range(2):
                        gj = bj*2 + j
                        gk = bk*2 + k
                        b_loc = j*2 + k
                        cc = V2[b_loc][t]
                        if cc:
                            V[gj*4 + gk][col] += cc * c
            for (bk, bi, c) in we:
                for k in range(2):
                    for i in range(2):
                        gk = bk*2 + k
                        gi = bi*2 + i
                        c_loc = k*2 + i
                        cc = W2[c_loc][t]
                        if cc:
                            W[gk*4 + gi][col] += cc * c
    return U, V, W


def _build_4x4x2():
    """(4x4)*(4x2) -> 4x2, rank 28 via 4 blocks of 2x2x2 Strassen."""
    U = [[0]*28 for _ in range(16)]
    V = [[0]*28 for _ in range(8)]
    W = [[0]*28 for _ in range(8)]
    U2, V2, W2 = _strassen_2x2()
    # 4 blocks: (ar,ac,br,cr,ck) = block row/col offsets in A, B, C
    blocks = [(0,0,0,0,0), (0,2,2,0,0), (2,0,0,2,0), (2,2,2,2,0)]
    for bi, (ar, ac, br, cr, ck) in enumerate(blocks):
        base = bi * 7
        for t in range(7):
            col = base + t
            for i in range(2):
                for j in range(2):
                    a_loc = i*2 + j
                    cc = U2[a_loc][t]
                    if cc:
                        U[(ar+i)*4 + (ac+j)][col] += cc
            for j in range(2):
                for k in range(2):
                    b_loc = j*2 + k
                    cc = V2[b_loc][t]
                    if cc:
                        V[(br+j)*2 + (ck+k)][col] += cc
            for k in range(2):
                for i in range(2):
                    c_loc = k*2 + i
                    cc = W2[c_loc][t]
                    if cc:
                        W[(ck+k)*4 + (cr+i)][col] += cc
    return U, V, W


def _build_4x1x2():
    """(4x1)*(1x2) -> 4x2, rank 8 (outer product / schoolbook)."""
    n, m, p = 4, 1, 2
    rank = n * m * p  # 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][c] = 1
                V[j*p + k][c] = 1
                W[k*n + i][c] = 1
                c += 1
    return U, V, W


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][c] = 1
                V[j*p + k][c] = 1
                W[k*n + i][c] = 1
                c += 1
    return U, V, W


def _combine(parts, n, m, p):
    """Combine block decompositions into a full (n,m,p) decomposition.
    Each part: (Us, Vs, Ws, (i0, j0, ii, jj, k0, kk))
      A block is ii x jj at offset (i0, j0)
      B block is jj x kk at offset (j0, k0)
      C block is ii x kk at offset (i0, k0)
    """
    total = sum(len(d[0][0]) for d in parts)
    U = [[0]*total for _ in range(n*m)]
    V = [[0]*total for _ in range(m*p)]
    W = [[0]*total for _ in range(p*n)]
    off = 0
    for (Us, Vs, Ws, (i0, j0, ii, jj, k0, kk)) in parts:
        nA = ii
        mA = jj
        nB = kk
        # U: A block rows i0..i0+ii-1, cols j0..j0+jj-1
        for r in range(nA * mA):
            i = r // mA
            j = r % mA
            for c in range(len(Us[0])):
                U[(i0+i)*m + (j0+j)][off+c] = Us[r][c]
        # V: B block rows j0..j0+jj-1, cols k0..k0+kk-1
        for r in range(mA * nB):
            j = r // nB
            k = r % nB
            for c in range(len(Vs[0])):
                V[(j0+j)*p + (k0+k)][off+c] = Vs[r][c]
        # W: C block rows k0..k0+kk-1, cols i0..i0+ii-1
        for r in range(nB * nA):
            k = r // nA
            i = r % nA
            for c in range(len(Ws[0])):
                W[(k0+k)*n + (i0+i)][off+c] = Ws[r][c]
        off += len(Us[0])
    return U, V, W


def _verify(U, V, W, n, m, p, trials=10, seed=12345):
    """Verify the decomposition exactly for random integer matrices."""
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
                    au = sum(U[i*m + j][t] * A[i][j] for j in range(m))
                    bu = sum(V[j*p + k][t] * B[j][k] for j in range(m))
                    val += W[k*n + i][t] * au * bu
                if val != C[i][k]:
                    return False
    return True


def _build_kronecker_tiling(n, m, p):
    """Tile 4x5 * 5x6 into 4x4x4 + 4x4x2 + 4x1x2 blocks.
    A is 4x5, split as [A11 (4x4) | A12 (4x1)]
    B is 5x6, split as [[B11 (4x4), B12 (4x2)], [B21 (1x4), B22 (1x2)]]
    C is 4x6, split as [C11 (4x4) | C12 (4x2)]
    C11 = A11 * B11
    C12 = A11 * B12 + A12 * B22
    """
    parts = []

    # C11 = A11 * B11: (4x4) * (4x4) -> 4x4, rank 49
    U1, V1, W1 = _build_4x4x4()
    parts.append((U1, V1, W1, (0, 0, 4, 4, 0, 4)))

    # C12 part 1: A11 * B12: (4x4) * (4x2) -> 4x2, rank 28
    U2, V2, W2 = _build_4x4x2()
    parts.append((U2, V2, W2, (0, 0, 4, 4, 4, 2)))

    # C12 part 2: A12 * B22: (4x1) * (1x2) -> 4x2, rank 8
    U3, V3, W3 = _build_4x1x2()
    parts.append((U3, V3, W3, (0, 4, 4, 1, 4, 2)))

    U, V, W = _combine(parts, n, m, p)
    return U, V, W


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 110  # safety margin before 120s limit

    try:
        # Primary: Kronecker-style tiling, rank 85
        U, V, W = _build_kronecker_tiling(n, m, p)
        rank = len(U[0])

        if time.time() > deadline - 20:
            U, V, W = _build_schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}

        if _verify(U, V, W, n, m, p, trials=10):
            if rank < n * m * p:
                return {"U": U, "V": V, "W": W}

        # Fallback: schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    except Exception:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
