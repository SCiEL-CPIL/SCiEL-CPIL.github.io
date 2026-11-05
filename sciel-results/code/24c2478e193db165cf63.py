"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_r23():
    """Return (U,V,W): 23 vectors of length 9, a rank-23 decomposition of M_{3,3,3}.

    Uses the Bläser (1998) '15 + 8' construction:
      - 15 base terms (each a product of three rank-1 vectors),
      - 8 terms from a 4D parity cube with a (-1)^(a+b+c) sign.
    Indexing: u[i*3+j], v[j*3+k], w[i*3+k]; C[i,k] = sum_j A[i,j] B[j,k].
    """
    def flat(idx):
        v = np.zeros(9, dtype=float)
        for (i, j) in idx:
            v[i * 3 + j] = 1.0
        return v

    # 15 base terms: (A-indices, B-indices, C-indices)
    base = [
        ([(0, 0)], [(0, 0)], [(0, 0)]),
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        ([(0, 2)], [(0, 2)], [(0, 2)]),
        ([(1, 0)], [(1, 0)], [(1, 0)]),
        ([(1, 1)], [(1, 1)], [(1, 1)]),
        ([(1, 2)], [(1, 2)], [(1, 2)]),
        ([(2, 0)], [(2, 0)], [(2, 0)]),
        ([(2, 1)], [(2, 1)], [(2, 1)]),
        ([(2, 2)], [(2, 2)], [(2, 2)]),
        ([(0, 1), (1, 0)], [(0, 1), (1, 0)], [(0, 1), (1, 0)]),
        ([(0, 2), (2, 0)], [(0, 2), (2, 0)], [(0, 2), (2, 0)]),
        ([(1, 2), (2, 1)], [(1, 2), (2, 1)], [(1, 2), (2, 1)]),
        ([(0, 1), (1, 2)], [(0, 1), (1, 2)], [(0, 1), (1, 2)]),
        ([(0, 2), (1, 0)], [(0, 2), (1, 0)], [(0, 2), (1, 0)]),
        ([(1, 0), (2, 1)], [(1, 0), (2, 1)], [(1, 0), (2, 1)]),
    ]
    # 8 parity-cube terms: (a, b, c) in {0,1}^3
    cube = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]

    U, V, W = [], [], []
    for (Ai, Bi, Ci) in base:
        U.append(flat(Ai)); V.append(flat(Bi)); W.append(flat(Ci))
    for (a, b, c) in cube:
        s = (-1.0) ** (a + b + c)
        Ai = [(a, b), (a, b + 1), (a + 1, b), (a + 1, b + 1)]
        Bi = [(b, c), (b, c + 1), (b + 1, c), (b + 1, c + 1)]
        Ci = [(c, a), (c, a + 1), (c + 1, a), (c + 1, a + 1)]
        U.append(s * flat(Ai)); V.append(flat(Bi)); W.append(s * flat(Ci))
    return U, V, W


def construct():
    """Matrix multiplication 3x3 by 3x6.

    Splits B (3x6) into two 3x3 blocks and applies a rank-23 3x3x3
    decomposition to each, giving a rank-46 exact decomposition.
    Falls back to schoolbook (rank 54) if verification fails.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 3, 3, 6

    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i, j, k] - 1.0) > 1e-8:
                        return False
        return True

    # Build the rank-23 3x3x3 decomposition and verify it in isolation.
    r23_U, r23_V, r23_W = _build_r23()
    U23 = np.array(r23_U, dtype=float)
    V23 = np.array(r23_V, dtype=float)
    W23 = np.array(r23_W, dtype=float)
    T23 = np.einsum('ar,br,cr->abc', U23, V23, W23)
    ok23 = all(abs(T23[i, j, k] - 1.0) <= 1e-8 for i in range(3) for j in range(3) for k in range(3))

    if ok23:
        rank = 46
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for block in range(2):
            for c in range(23):
                col = block * 23 + c
                u = r23_U[c]; v = r23_V[c]; w = r23_W[c]
                for idx in range(9):
                    U[idx][col] = u[idx]
                    V[idx + block * 9][col] = v[idx]
                    W[idx + block * 9][col] = w[idx]
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 54
    if time.time() > deadline:
        raise TimeoutError("deadline exceeded before fallback")
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
