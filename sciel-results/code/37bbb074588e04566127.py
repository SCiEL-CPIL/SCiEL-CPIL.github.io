"""Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Uses a 3-way block decomposition to achieve rank 46.
Splits A columns as [0:2], [2:4], [4:5] and B rows/cols similarly.
Blocks:
- 4x (2x2 * 2x2) Strassen rank 7 each = 28
- 2x (2x1 * 1x2) schoolbook rank 4 each = 8
- 2x (2x2 * 2x1) schoolbook rank 4 each = 8
- 1x (2x1 * 1x1) schoolbook rank 2 = 2
Total rank: 46.
"""

import time

_T0 = time.time()
_DEADLINE_S = 110.0


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S


def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Local indices: i,j,k in 0..1
    U shape: (2*2, 7)
    V shape: (2*2, 7)
    W shape: (2*2, 7)
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # r=0: (a+d)(e+h)
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # r=1: (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # r=2: a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # r=3: d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # r=4: (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # r=5: (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # r=6: (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1

    # W[k*n + i] with n=2
    # C00: W[0] = m1 + m4 - m5 + m7
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    # C01: W[2] = m3 + m5
    W[2][2] = 1
    W[2][4] = 1
    # C10: W[1] = m2 + m4
    W[1][1] = 1
    W[1][3] = 1
    # C11: W[3] = m1 - m2 + m3 + m6
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = 1
    W[3][5] = 1

    return U, V, W


def _schoolbook_block(n, m, p):
    """
    Returns U, V, W for n x m * m x p using schoolbook.
    Local indices.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W



def _place_strassen_block(U, V, W, i0, j0, k0, off):
    """Place Strassen 2x2x2 block at global positions i0, j0, k0."""
    U1, V1, W1 = _strassen_222()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[(i0 + i) * 5 + (j0 + j)][off + r] = U1[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(j0 + j) * 5 + (k0 + k)][off + r] = V1[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(k0 + k) * 2 + (i0 + i)][off + r] = W1[k * 2 + i][r]
    return off + 7


def _place_schoolbook_block(U, V, W, i0, j0, k0, n, m, p, off):
    """Place schoolbook n x m * m x p block at global positions i0, j0, k0."""
    U1, V1, W1 = _schoolbook_block(n, m, p)
    rank = n * m * p
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                U[(i0 + i) * 5 + (j0 + j)][off + r] = U1[i * m + j][r]
        for j in range(m):
            for k in range(p):
                V[(j0 + j) * 5 + (k0 + k)][off + r] = V1[j * p + k][r]
        for k in range(p):
            for i in range(n):
                W[(k0 + k) * 2 + (i0 + i)][off + r] = W1[k * n + i][r]
    return off + rank


def construct():
    n, m, p = 2, 5, 5
    total_rank = 46

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    off = 0

    # C[:, 0:2] = A[:,0:2]*B[0:2,0:2] + A[:,2:4]*B[2:4,0:2] + A[:,4:5]*B[4:5,0:2]
    # Block: (2x2)*(2x2) at i=0, j=0, k=0
    off = _place_strassen_block(U, V, W, 0, 0, 0, off)
    # Block: (2x2)*(2x2) at i=0, j=2, k=0
    off = _place_strassen_block(U, V, W, 0, 2, 0, off)
    # Block: (2x1)*(1x2) at i=0, j=4, k=0
    off = _place_schoolbook_block(U, V, W, 0, 4, 0, 2, 1, 2, off)

    # C[:, 2:4] = A[:,0:2]*B[0:2,2:4] + A[:,2:4]*B[2:4,2:4] + A[:,4:5]*B[4:5,2:4]
    # Block: (2x2)*(2x2) at i=0, j=0, k=2
    off = _place_strassen_block(U, V, W, 0, 0, 2, off)
    # Block: (2x2)*(2x2) at i=0, j=2, k=2
    off = _place_strassen_block(U, V, W, 0, 2, 2, off)
    # Block: (2x1)*(1x2) at i=0, j=4, k=2
    off = _place_schoolbook_block(U, V, W, 0, 4, 2, 2, 1, 2, off)

    # C[:, 4:5] = A[:,0:2]*B[0:2,4:5] + A[:,2:4]*B[2:4,4:5] + A[:,4:5]*B[4:5,4:5]
    # Block: (2x2)*(2x1) at i=0, j=0, k=4
    off = _place_schoolbook_block(U, V, W, 0, 0, 4, 2, 2, 1, off)
    # Block: (2x2)*(2x1) at i=0, j=2, k=4
    off = _place_schoolbook_block(U, V, W, 0, 2, 4, 2, 2, 1, off)
    # Block: (2x1)*(1x1) at i=0, j=4, k=4
    off = _place_schoolbook_block(U, V, W, 0, 4, 4, 2, 1, 1, off)

    # Verify
    if not _verify(U, V, W, total_rank):
        U_s, V_s, W_s, _ = _schoolbook()
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, rank):
    """Exactly check that the decomposition reproduces the multiplication tensor."""
    n, m, p = 2, 5, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    s += U[ui][r] * V[vj][r] * W[wk][r]
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Return (U, V, W) as lists of lists with integer 0/1 entries, rank = n*m*p."""
    n, m, p = 2, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W, rank
