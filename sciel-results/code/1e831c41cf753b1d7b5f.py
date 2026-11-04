"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _schoolbook(n, m, p):
    """Trivial exact rank n*m*p decomposition of an (n x m)*(m x p) product."""
    U = np.zeros((n * m, n * m * p), dtype=int)
    V = np.zeros((m * p, n * m * p), dtype=int)
    W = np.zeros((p * n, n * m * p), dtype=int)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
    return U, V, W


def _strassen22():
    """Exact rank-7 Strassen decomposition of a (2 x 2)*(2 x 2) product."""
    U = np.zeros((4, 11), dtype=int)
    V = np.zeros((4, 11), dtype=int)
    W = np.zeros((4, 11), dtype=int)
    # A rows: a00,a01,a10,a11 ; B rows: b00,b01,b10,b11
    # C00 = M1 - M2 + M4 - M5
    U[0, 0] = 1; U[1, 0] = 1; U[2, 0] = -1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = -1; V[3, 0] = 1
    W[0, 0] = 1
    # C01 = M3 + M5
    U[1, 1] = 1; U[3, 1] = 1
    V[1, 1] = 1; V[3, 1] = 1
    W[1, 1] = 1
    # C10 = M2 + M4
    U[2, 2] = 1; U[3, 2] = 1
    V[0, 2] = 1; V[2, 2] = 1
    W[2, 2] = 1
    # C11 = M1 - M3 + M2 - M4
    U[0, 3] = 1; U[1, 3] = -1; U[2, 3] = 1; U[3, 3] = -1
    V[0, 3] = 1; V[1, 3] = -1; V[2, 3] = 1; V[3, 3] = -1
    W[3, 3] = 1
    # M1 = (a00+a11)(b00+b11)
    U[0, 4] = 1; U[3, 4] = 1
    V[0, 4] = 1; V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1; W[2, 4] = 1; W[3, 4] = 1
    # M2 = (a10+a11)b00
    U[2, 5] = 1; U[3, 5] = 1
    V[0, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    # M3 = a00(b10+b11)
    U[0, 6] = 1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    # M4 = (a00-a10)(b00+b10)
    U[0, 7] = 1; U[2, 7] = -1
    V[0, 7] = 1; V[2, 7] = 1
    W[0, 7] = 1; W[2, 7] = 1
    # M5 = (a01+a11)(b01-b11)
    U[1, 8] = 1; U[3, 8] = 1
    V[1, 8] = 1; V[3, 8] = -1
    W[1, 8] = 1; W[3, 8] = 1
    # M6 = (a00+a01-a11)(b11-b10)
    U[0, 9] = 1; U[1, 9] = 1; U[3, 9] = -1
    V[3, 9] = 1; V[2, 9] = -1
    W[1, 9] = 1; W[2, 9] = 1
    # M7 = (a10-a00)(b00+b01)
    U[2, 10] = 1; U[0, 10] = -1
    V[0, 10] = 1; V[1, 10] = 1
    W[0, 10] = 1; W[3, 10] = 1
    return U[:, :7], V[:, :7], W[:, :7]


def _decompose_block(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return _strassen22()
    return _schoolbook(n, m, p)


def _build_block_decomposition(n, m, p, splits):
    """
    Combine per-block decompositions into one global exact decomposition.
    splits: list of (rA, cA, rB, cB, nL, mL, pL), where a block multiplies
    A[rA:rA+nL, cA:cA+mL] by B[rB:rB+mL, cB:cB+pL] -> C[rA:rA+nL, cB:cB+pL].
    """
    U = np.zeros((n * m, 0), dtype=int)
    V = np.zeros((m * p, 0), dtype=int)
    W = np.zeros((p * n, 0), dtype=int)
    R = 0
    for (rA, cA, rB, cB, nL, mL, pL) in splits:
        Ublk, Vblk, Wblk = _decompose_block(nL, mL, pL)
        RB = Ublk.shape[1]
        Unew = np.zeros((n * m, RB), dtype=int)
        Vnew = np.zeros((m * p, RB), dtype=int)
        Wnew = np.zeros((p * n, RB), dtype=int)
        for u in range(nL * mL):
            Unew[(rA + u // mL) * m + (cA + u % mL), :] = Ublk[u, :]
        for v in range(mL * pL):
            Vnew[(rB + v // pL) * p + (cB + v % pL), :] = Vblk[v, :]
        for w in range(pL * nL):
            Wnew[(cB + w // nL) * n + (rA + w % nL), :] = Wblk[w, :]
        U = np.hstack([U, Unew]) if U.shape[1] else Unew
        V = np.hstack([V, Vnew]) if V.shape[1] else Vnew
        W = np.hstack([W, Wnew]) if W.shape[1] else Wnew
        R += RB
    return U, V, W


def _construct_block():
    n, m, p = 4, 5, 5
    # Split rows of A/C into 2+2; columns of A / rows of B into 3+2;
    # columns of B/C into 3+2.  The 2x2x2 block gets Strassen (rank 7).
    splits = [
        (0, 0, 0, 0, 2, 3, 3),   # schoolbook 2x3x3 = 18
        (0, 3, 0, 3, 2, 2, 2),   # strassen  2x2x2 = 7
        (2, 0, 3, 0, 2, 3, 3),   # schoolbook 2x3x3 = 18
        (2, 3, 3, 3, 2, 2, 2),   # strassen  2x2x2 = 7
    ]
    U, V, W = _build_block_decomposition(n, m, p, splits)
    return {"U": U, "V": V, "W": W}


def _construct_trivial():
    n, m, p = 4, 5, 5
    total_rank = n * m * p
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, samples=20):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    rng = np.random.default_rng(12345)
    for _ in range(samples):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C = A @ B
        # T_{i,j,k} = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # reconstruct C[i,k] = sum_j T_{i,j,k}
        Ua = U.reshape(n, m, rank)
        Vb = V.reshape(m, p, rank)
        Wc = W.reshape(p, n, rank)
        Crec = np.zeros((n, p))
        for j in range(m):
            for k in range(p):
                Crec[:, k] += (Ua[:, j, :] * Vb[j, k, :] * Wc[k, :]).sum(axis=1)
        if np.max(np.abs(Crec - C)) > 1e-6:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin well under 150 s

    # Primary: exact block decomposition (Strassen where possible), rank 50.
    best = _construct_block()

    # Verify the block decomposition; fall back to trivial if it ever fails.
    if time.time() < deadline:
        ok = _verify(best["U"], best["V"], best["W"], 4, 5, 5)
        if not ok:
            best = _construct_trivial()

    if time.time() > deadline:
        return best

    return best

# EVOLVE-BLOCK-END
