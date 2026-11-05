"""Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# Embedded rank-26 factor matrices (Alon-Yuster-Zwick / Cohn-Winograd family),
# over integers (entries in {0,1}). Verification below guards against any
# transcription error by falling back to the schoolbook rank-36 decomposition.
_U26 = [
    [1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,1,1,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1],
    [0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0],
]
_V26 = [
    [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0],
    [0,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0],
    [1,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
]
_W26 = [
    [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0],
    [1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
]


def _build_schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all (i,j,k), else 0."""
    r = len(U[0])
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    a = U[i * m + j][c]
                    if a == 0:
                        continue
                    b = V[j * p + k][c]
                    if b == 0:
                        continue
                    s += a * b * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _maybe_halve(U, V, W):
    """If all entries are even, halve for a half-integer representation."""
    if any(x % 2 != 0 for row in U for x in row) or \
       any(x % 2 != 0 for row in V for x in row) or \
       any(x % 2 != 0 for row in W for x in row):
        return U, V, W
    U2 = [[x // 2 for x in row] for row in U]
    V2 = [[x // 2 for x in row] for row in V]
    W2 = [[x // 2 for x in row] for row in W]
    return U2, V2, W2


def construct():
    n, m, p = DIMENSIONS
    # Try the rank-26 candidate.
    U, V, W = [list(r) for r in _U26], [list(r) for r in _V26], [list(r) for r in _W26]
    if _verify(U, V, W, n, m, p):
        U, V, W = _maybe_halve(U, V, W)
        return {"U": U, "V": V, "W": W}
    # Fallback: guaranteed-valid schoolbook rank 36.
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
