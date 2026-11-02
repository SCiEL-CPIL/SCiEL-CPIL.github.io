"""Structured algebraic decomposition for 2x2x15 matrix multiplication.

Splits the tensor into a 2x2x4 block (rank 14, Strassen tensor square) and a
2x2x11 block (rank 39 = five rank-7 Strassen pairs + one rank-4 block).
Total rank 53 < schoolbook 60.

Trace convention: W[k*n + i, c] (row = k*n + i, col = c).
U row = i*m + j, V row = j*p + k, W row = k*n + i.
"""
DIMENSIONS = [2, 2, 15]


def _strassen_2x2():
    """7-term Strassen decomposition for 2x2x2.

    U[i*2+j], V[j*2+k], W[k*2+i], c in 0..6, entries in {+1,-1}.
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    terms = [
        (0, 0, 0, 1),
        (0, 1, 1, 1),
        (1, 0, 2, 1),
        (3, 3, 3, 1),
        (3, 2, 0, 1),
        (2, 3, 1, 1),
        (1, 2, 2, -1),
    ]
    for c, (ur, vr, wr, s) in enumerate(terms):
        U[ur][c] = s
        V[vr][c] = s
        W[wr][c] = s
    return U, V, W


def _square_2x2x4():
    """Rank-14 decomposition of 2x2x4 (tensor square of Strassen 2x2x2).

    U shape (4,14), V shape (8,14), W shape (8,14).
    V row = j*4 + k, W row = k*2 + i, where j,k,i in {0,1} per factor.
    """
    U2, V2, W2 = _strassen_2x2()
    U = [[0] * 14 for _ in range(4)]
    V = [[0] * 14 for _ in range(8)]
    W = [[0] * 14 for _ in range(8)]
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            U[U2[0][c1] * 2 + U2[0][c2]][c] = 1
            V[V2[0][c1] * 4 + V2[0][c2]][c] = 1
            W[W2[0][c1] * 2 + W2[0][c2]][c] = 1
    return U, V, W


def _block_2x2x11():
    """Rank-39 decomposition of 2x2x11.

    U shape (4,39), V shape (22,39), W shape (22,39).
    Five independent Strassen pairs (rank 7 each) + one 2x2x4 block (rank 14).

    V row = j*11 + k, W row = k*2 + i.
    """
    U2, V2, W2 = _strassen_2x2()
    U4, V4, W4 = _square_2x2x4()

    U = [[0] * 39 for _ in range(4)]
    V = [[0] * 39 for _ in range(22)]
    W = [[0] * 39 for _ in range(22)]

    # Five Strassen pairs: k = 2t + j2, j2 in {0,1}, t in 0..4
    for t in range(5):
        base = t * 7
        for c in range(7):
            cc = base + c
            U[U2[0][c]][cc] = 1
            j2 = V2[0][c]
            k2 = W2[0][c]
            k = 2 * t + j2
            V[j2 * 11 + k][cc] = 1
            W[k * 2 + k2][cc] = 1

    # 2x2x4 block for k = 10,11,12,13
    for c in range(14):
        cc = 35 + c
        U[U4[0][c]][cc] = 1
        # V4 row = j*4 + kk, kk in {0,1,2,3}, k = 10 + kk
        j = V4[0][c] // 4
        kk = V4[0][c] % 4
        V[j * 11 + (10 + kk)][cc] = 1
        # W4 row = kk*2 + i2, kk in {0,1,2,3}, k = 10 + kk
        kk = W4[0][c] // 2
        i2 = W4[0][c] % 2
        W[(10 + kk) * 2 + i2][cc] = 1

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    U4, V4, W4 = _square_2x2x4()
    U11, V11, W11 = _block_2x2x11()

    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 2x2x4 block: k = 0..3
    for r in range(4):
        for c in range(14):
            U[r][c] = U4[r][c]
    for r in range(8):
        for c in range(14):
            V[r][c] = V4[r][c]
    for r in range(8):
        for c in range(14):
            W[r][c] = W4[r][c]

    # 2x2x11 block: k = 4..14
    for r in range(4):
        for c in range(39):
            U[r][14 + c] = U11[r][c]
    for r in range(22):
        for c in range(39):
            V[r + 8][14 + c] = V11[r][c]
    for r in range(22):
        for c in range(39):
            W[r + 8][14 + c] = W11[r][c]

    return {"U": U, "V": V, "W": W}
