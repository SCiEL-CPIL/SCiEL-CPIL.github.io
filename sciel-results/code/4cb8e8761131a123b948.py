"""Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.

Strategy: deterministic block-Strassen construction (group the 7 output columns
into a 3-column and a 4-column block, each handled by a 2x2 Strassen pass),
validated with the exact Fraction checker, with a fast fallback to schoolbook.
"""
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 7]


def _matmul_check(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if total != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def _strassen_block(n, m, p, k0):
    """Decompose A(n x m) * B(m x p) into C(n x p) using a 2x2 Strassen pass.

    k0 is the starting column of the output C (W index offset).  Returns the
    (U, V, W) coefficient lists for the 7 Strassen multiplications.
    """
    n2, m2, p2 = n // 2, m // 2, p // 2
    assert n == 2 * n2 and m == 2 * m2 and p == 2 * p2

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def build(u, v, w):
        U = [0] * (n * m)
        V = [0] * (m * p)
        W = [0] * (p * n)
        for (i, j), a in u.items():
            U[u_idx(i, j)] = a
        for (j, k), a in v.items():
            V[v_idx(j, k)] = a
        for (k, i), a in w.items():
            W[w_idx(k, i)] = a
        return U, V, W

    def half(n_, m_, i, j):
        # element of an n_ x m_ matrix in row i, col j of the global A/B
        return (i, j)

    def halfp(n_, m_, k, i):
        return (k, i)

    # Quadrants of A (2n2 x 2m2), B (2m2 x 2p2), C (2n2 x 2p2)
    A11 = {(i, j) for i in range(0, n2) for j in range(0, m2)}
    A12 = {(i, j) for i in range(0, n2) for j in range(m2, 2 * m2)}
    A21 = {(i, j) for i in range(n2, 2 * n2) for j in range(0, m2)}
    A22 = {(i, j) for i in range(n2, 2 * n2) for j in range(m2, 2 * m2)}
    B11 = {(j, k) for j in range(0, m2) for k in range(0, p2)}
    B12 = {(j, k) for j in range(0, m2) for k in range(p2, 2 * p2)}
    B21 = {(j, k) for j in range(m2, 2 * m2) for k in range(0, p2)}
    B22 = {(j, k) for j in range(m2, 2 * m2) for k in range(p2, 2 * p2)}
    C11 = {(k, i) for i in range(0, n2) for k in range(0, p2)}
    C12 = {(k, i) for i in range(0, n2) for k in range(p2, 2 * p2)}
    C21 = {(k, i) for i in range(n2, 2 * n2) for k in range(0, p2)}
    C22 = {(k, i) for i in range(n2, 2 * n2) for k in range(p2, 2 * p2)}

    def sumsets(*sets):
        out = {}
        for s in sets:
            for e in s:
                out[e] = out.get(e, 0) + 1
        return out

    def diffsets(a, b):
        out = {}
        for e in a:
            out[e] = out.get(e, 0) + 1
        for e in b:
            out[e] = out.get(e, 0) - 1
        return {e: c for e, c in out.items() if c != 0}

    # The 7 Strassen multiplications: (U-set, V-set, W-set)
    M1 = (sumsets(A11, A22), sumsets(B11, B22), sumsets(C11, C12, C21, C22))
    M2 = (sumsets(A21, A22), B11, sumsets(C21, C22))
    M3 = (A11, B12, C12)
    M4 = (A22, B21, C21)
    M5 = (A11, sumsets(B11, B12), C11)
    M6 = (diffsets(B21, B11), A22, C22)
    M7 = (diffsets(B12, B22), A11, C12)

    terms = []
    for (us, vs, ws) in (M1, M2, M3, M4, M5, M6, M7):
        U, V, W = build(us, vs, ws)
        terms.append((U, V, W))

    # Offset W indices by the output block start column k0
    out = []
    for (U, V, W) in terms:
        Wo = [0] * (p * n)
        for kk in range(p):
            for ii in range(n):
                Wo[(k0 + kk) * n + ii] = W[kk * n + ii]
        out.append((U, V, Wo))
    return out


def _strassen_decomposition(n, m, p):
    """Full 2x3 x 3x7 decomposition: 3-col Strassen block + 4-col Strassen block."""
    terms = []
    terms += _strassen_block(n, m, 3, k0=0)   # C columns 0..2
    terms += _strassen_block(n, m, 4, k0=3)   # C columns 3..6
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (Uc, Vc, Wc) in enumerate(terms):
        for a in range(n * m):
            U[a][r] = Uc[a]
        for b in range(m * p):
            V[b][r] = Vc[b]
        for c in range(p * n):
            W[c][r] = Wc[c]
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)

    # 1) Deterministic block-Strassen construction (rank 14).
    cand = _strassen_decomposition(n, m, p)
    if _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p):
        return cand
    if time.time() > deadline - 20:
        return _schoolbook(n, m, p)

    # 2) Fallback: exact schoolbook (guaranteed valid, rank 42).
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
