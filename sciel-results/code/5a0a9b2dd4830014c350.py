"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                          for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
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


def _polish(U, V, W, n, m, p, max_iter=20000):
    """Polish a candidate decomposition to exact integer/half-integer entries.

    Uses exact rational arithmetic (fractions.Fraction) so the final check is
    exact, not floating point. If the tensor contract equals 1 everywhere the
    decomposition is valid; entries are rounded to the nearest half-integer.
    Returns (U_list, V_list, W_list, rank) or None.
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    target = np.ones((n, m, p), dtype=float)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)

    for _ in range(max_iter):
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(T - target))
        if err < 1e-9:
            break
        # gradients of 0.5*sum (T-1)^2
        gU = np.einsum('ijc,jkc,kic->ijc', (T - target), V3, W3)
        gV = np.einsum('ijc,jkc,kic->jkc', (T - target), U3, W3)
        gW = np.einsum('ijc,jkc,kic->kic', (T - target), U3, V3)
        gU = gU.reshape(n * m, rank)
        gV = gV.reshape(m * p, rank)
        gW = gW.reshape(p * n, rank)
        step = 0.1 * (1.0 + err)
        U -= step * gU
        V -= step * gV
        W -= step * gW
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)

    # Exact verification with rationals
    Uf = [[Fraction(round(float(x), 9)) for x in row] for row in U.tolist()]
    Vf = [[Fraction(round(float(x), 9)) for x in row] for row in V.tolist()]
    Wf = [[Fraction(round(float(x), 9)) for x in row] for row in W.tolist()]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                if s != 1:
                    return None

    # Round to nearest half-integer and re-verify exactly
    def _round_half(x):
        return Fraction(round(2.0 * float(x)), 2)

    Uq = [[_round_half(x) for x in row] for row in U.tolist()]
    Vq = [[_round_half(x) for x in row] for row in V.tolist()]
    Wq = [[_round_half(x) for x in row] for row in W.tolist()]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uq[i * m + j][c] * Vq[j * p + k][c] * Wq[k * n + i][c]
                if s != 1:
                    return None
    return Uq, Vq, Wq, rank


def _strassen_3x4x6(n, m, p):
    """Structured Strassen-style ansatz for 3x4x6.

    Partitions A (3x4) and B (4x6) into blocks and uses 'sum' products that
    cancel in the output (the core Strassen trick), reducing rank below 72.
    Returns (U, V, W) as float lists of shapes (n*m, r), (m*p, r), (p*n, r).
    """
    n, m, p = 3, 4, 6
    Arows = [(0, 1), (2, 3), (4, 5)]
    Bcols = [(0, 1), (2, 3), (4, 5)]
    a = [[Arows[i][0], Arows[i][1]] for i in range(3)]
    b = [[Bcols[k][0], Bcols[k][1]] for k in range(3)]

    def Aterm(i, j):
        if j == 0:
            return [(0, 1), (1, 1)]
        elif j == 1:
            return [(0, 2), (1, 2)]
        else:
            return [(0, 3), (1, 3)]

    def Bterm(j, k):
        if k == 0:
            return [(0, 0), (1, 0)]
        elif k == 1:
            return [(0, 1), (1, 1)]
        else:
            return [(0, 2), (1, 2)]

    U, V, W = [], [], []

    def add(term, c, s):
        U.append(term)
        V.append(c)
        W.append(s)

    for i in range(3):
        for k in range(3):
            # 4 direct block products (no cancellation)
            for j in range(2):
                add(Aterm(i, j), Bterm(j, k), [(k, i)])
            # 3 Strassen 'sum' products, each cancels in the output
            add(Aterm(i, 0) + Aterm(i, 1), Bterm(0, k) + Bterm(1, k), [(k, i), (k, i)])
            add(Aterm(i, 0), Bterm(0, k) + Bterm(1, k), [(k, i), (k, i)])
            add(Aterm(i, 1), Bterm(0, k) + Bterm(1, k), [(k, i), (k, i)])

    U = np.array(U, dtype=float).reshape(n * m, len(U))
    V = np.array(V, dtype=float).reshape(m * p, len(V))
    W = np.array(W, dtype=float).reshape(p * n, len(W))
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, leave 5s margin

    # Fallback: schoolbook (rank n*m*p), always valid.
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = n * m * p

    # Structured Strassen-style candidate, then polish to exact.
    if time.time() < deadline - 15.0:
        U0, V0, W0 = _strassen_3x4x6(n, m, p)
        res = _polish(U0, V0, W0, n, m, p, max_iter=40000)
        if res is not None:
            Uq, Vq, Wq, r = res
            U_list = [[float(x) for x in row] for row in Uq]
            V_list = [[float(x) for x in row] for row in Vq]
            W_list = [[float(x) for x in row] for row in Wq]
            if _verify(U_list, V_list, W_list, n, m, p):
                if r < best_rank:
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r

    return best
# EVOLVE-BLOCK-END
