"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    # Rank 50: 6 Strassen blocks (2x2x2) for k-pairs (0,1),(2,3),(4,5) x j-pairs (0,1),(2,3)
    # + 4 schoolbook 2x2x1 blocks for k=6, j-pairs (0,1),(2,3)
    # 6*7 + 2*4 = 42 + 8 = 50
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    # Strassen 2x2x2 block: multiplies [[a,b],[c,d]] * [[e,f],[g,h]]
    # 7 scalar multiplications, each contributing to output entries
    def _strassen_block(i0, j0, k0):
        nonlocal col
        # Input indices
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1
        e = j0 * p + k0
        f = j0 * p + k0 + 1
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + k0 + 1
        # Output indices (W[k*n + i, r])
        o00 = k0 * n + i0
        o01 = (k0 + 1) * n + i0
        o10 = k0 * n + i0 + 1
        o11 = (k0 + 1) * n + i0 + 1

        # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[o00, col] += 1; W[o11, col] += 1
        col += 1

        # M2 = (c+d)e -> C10 += M2, C11 -= M2
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[o10, col] += 1; W[o11, col] -= 1
        col += 1

        # M3 = a(f-h) -> C01 += M3, C11 += M3
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[o01, col] += 1; W[o11, col] += 1
        col += 1

        # M4 = d(g-e) -> C00 += M4, C10 += M4
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[o00, col] += 1; W[o10, col] += 1
        col += 1

        # M5 = (a+b)h -> C00 -= M5, C01 += M5
        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[o00, col] -= 1; W[o01, col] += 1
        col += 1

        # M6 = (c-a)(e+f) -> C11 += M6
        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[o11, col] += 1
        col += 1

        # M7 = (b-d)(g+h) -> C00 += M7
        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[o00, col] += 1
        col += 1

    # Schoolbook 2x2x1 block: [[a,b],[c,d]] * [[e],[g]] -> [[a*e+b*g],[c*e+d*g]]
    def _sb_block(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        o0 = k0 * n + i0
        o1 = k0 * n + i0 + 1

        U[a, col] += 1; V[e, col] += 1; W[o0, col] += 1; col += 1
        U[b, col] += 1; V[g, col] += 1; W[o0, col] += 1; col += 1
        U[c, col] += 1; V[e, col] += 1; W[o1, col] += 1; col += 1
        U[d, col] += 1; V[g, col] += 1; W[o1, col] += 1; col += 1

    # Apply Strassen for k-pairs (0,1),(2,3),(4,5) and j-pairs (0,1),(2,3)
    for jp in (0, 2):
        for kp in (0, 2, 4):
            _strassen_block(0, jp, kp)
        # Schoolbook for k=6
        _sb_block(0, jp, 6)

    if col != rank:
        return _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
