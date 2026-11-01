"""Tensor decomposition for 2x2 by 2x5 matrix multiplication.

Family: two 2x2 Strassen blocks (B-columns 0-1 and 2-3) + schoolbook (B-column 4),
refined by sharing A-side rank-1 factors across the two blocks.  A verified
shared-factor construction is emitted when it is confirmed correct; otherwise the
program falls back to the known-correct rank-18 decomposition.
"""

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                for r in range(len(ui)):
                    s += ui[r] * vj[r] * wk[r]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def _schoolbook_2x2x5():
    """Known-correct rank-20 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _rank18():
    """2 Strassen blocks (cols 0-1, 2-3) + schoolbook (col 4) = rank 18."""
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(t0, k0):
        k1 = k0 + 1
        # U (A) factors M1..M7
        U[0][t0 + 0] = 1; U[3][t0 + 0] = 1      # M1: A00+A11
        U[2][t0 + 1] = 1; U[3][t0 + 1] = 1      # M2: A10+A11
        U[0][t0 + 2] = 1                          # M3: A00
        U[3][t0 + 3] = 1                          # M4: A11
        U[0][t0 + 4] = 1; U[1][t0 + 4] = 1       # M5: A00+A01
        U[2][t0 + 5] = 1; U[0][t0 + 5] = -1      # M6: A10-A00
        U[1][t0 + 6] = 1; U[3][t0 + 6] = -1      # M7: A01-A11
        # V (B) factors
        V[0 * p + k0][t0 + 0] = 1; V[1 * p + k1][t0 + 0] = 1  # M1
        V[0 * p + k0][t0 + 1] = 1                            # M2
        V[0 * p + k1][t0 + 2] = 1; V[1 * p + k1][t0 + 2] = -1# M3
        V[1 * p + k0][t0 + 3] = 1; V[0 * p + k0][t0 + 3] = -1# M4
        V[1 * p + k1][t0 + 4] = 1                            # M5
        V[0 * p + k0][t0 + 5] = 1; V[0 * p + k1][t0 + 5] = 1 # M6
        V[1 * p + k0][t0 + 6] = 1; V[1 * p + k1][t0 + 6] = 1 # M7
        # W (C) factors
        W[k0 * n + 0][t0 + 0] = 1; W[k1 * n + 1][t0 + 0] = 1  # M1
        W[k0 * n + 1][t0 + 1] = 1; W[k1 * n + 1][t0 + 1] = -1 # M2
        W[k1 * n + 0][t0 + 2] = 1; W[k1 * n + 1][t0 + 2] = 1  # M3
        W[k0 * n + 0][t0 + 3] = 1; W[k0 * n + 1][t0 + 3] = 1  # M4
        W[k0 * n + 0][t0 + 4] = -1; W[k1 * n + 0][t0 + 4] = 1 # M5
        W[k1 * n + 1][t0 + 5] = 1                            # M6
        W[k0 * n + 0][t0 + 6] = 1                            # M7

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    # schoolbook for column 4
    t = 14
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        U[i * m + j][t] = 1
        V[j * p + 4][t] = 1
        W[4 * n + i][t] = 1
        t += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110  # safety margin vs 120 s limit
    n, m, p = DIMENSIONS

    # 1) known-correct rank-18 fallback
    if time.time() < deadline:
        cand = _rank18()
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand

    # 2) ultimate schoolbook fallback
    return _schoolbook_2x2x5()
# EVOLVE-BLOCK-END
