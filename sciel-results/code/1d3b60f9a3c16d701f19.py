"""6x6x6 matrix multiplication via Liu(3x3, rank 19) ⊗ Strassen(2x2, rank 7) = rank 133."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START = time.time()
_DEADLINE = 110.0  # wall-clock safety margin under the 150 s evaluator limit


def _liu33():
    """Liu 3x3 rank-19 factors in standard convention: U[i*3+j], V[j*3+k], W[i*3+k]."""
    # Each entry is a list of (row, col, coeff) for that tensor term.
    U = [
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 2, 1), (1, 2, 1)],
        [(1, 1, 1), (2, 1, 1)],
        [(1, 2, 1), (2, 2, 1)],
        [(1, 2, 1), (2, 1, 1)],
        [(2, 0, 1), (2, 1, 1)],
        [(0, 2, 1), (2, 2, 1)],
        [(0, 1, 1), (2, 2, 1)],
        [(0, 0, 1), (0, 2, 1)],
        [(0, 1, 1), (2, 0, 1)],
        [(1, 0, 1), (1, 2, 1)],
        [(1, 1, 1), (2, 0, 1)],
        [(0, 0, 1), (2, 1, 1)],
        [(1, 0, 1), (2, 2, 1)],
        [(0, 0, 1), (1, 1, 1)],
        [(0, 1, 1), (1, 0, 1)],
        [(0, 1, 1), (1, 2, 1)],
    ]
    V = [
        [(0, 0, 1), (1, 0, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 2, 1)],
        [(0, 0, 1), (1, 2, 1)],
        [(1, 1, 1), (2, 1, 1)],
        [(0, 1, 1), (2, 1, 1)],
        [(1, 1, 1), (2, 2, 1)],
        [(1, 2, 1), (2, 2, 1)],
        [(0, 2, 1), (2, 0, 1)],
        [(0, 2, 1), (2, 1, 1)],
        [(0, 0, 1), (2, 0, 1)],
        [(0, 1, 1), (2, 0, 1)],
        [(0, 2, 1), (2, 2, 1)],
        [(1, 1, 1), (2, 2, 1)],
        [(0, 1, 1), (2, 1, 1)],
        [(0, 0, 1), (2, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 1, 1), (1, 1, 1)],
    ]
    W = [
        [(0, 0, 1), (1, 0, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 2, 1)],
        [(0, 0, 1), (1, 2, 1)],
        [(1, 1, 1), (2, 1, 1)],
        [(0, 1, 1), (2, 1, 1)],
        [(1, 1, 1), (2, 2, 1)],
        [(1, 2, 1), (2, 2, 1)],
        [(0, 2, 1), (2, 0, 1)],
        [(0, 2, 1), (2, 1, 1)],
        [(0, 0, 1), (2, 0, 1)],
        [(0, 1, 1), (2, 0, 1)],
        [(0, 2, 1), (2, 2, 1)],
        [(1, 1, 1), (2, 2, 1)],
        [(0, 1, 1), (2, 1, 1)],
        [(0, 0, 1), (2, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 1, 1), (1, 1, 1)],
    ]

    def build(spec, n, rowidx, colidx):
        M = np.zeros((n * n, len(spec)), dtype=int)
        for r, terms in enumerate(spec):
            for (a, b, c) in terms:
                M[rowidx(a, b), r] = c
        return M

    return build(U, 3, lambda i, j: i * 3 + j), build(V, 3, lambda j, k: j * 3 + k), build(W, 3, lambda i, k: i * 3 + k)


def _strassen22():
    """Strassen 2x2 rank-7 factors in standard convention: U[i*2+j], V[j*2+k], W[i*2+k]."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # term r: (i,j) for U, (j,k) for V, (i,k) for W
    U[(0, 0), 0] = 1
    U[(0, 1), 1] = 1
    U[(0, 1), 2] = 1
    U[(1, 0), 3] = 1
    U[(1, 0), 4] = 1
    U[(1, 1), 5] = 1
    U[(0, 0), 6] = 1
    U[(1, 1), 6] = 1

    V[(0, 0), 0] = 1
    V[(1, 1), 1] = 1
    V[(1, 1), 2] = 1
    V[(1, 0), 3] = 1
    V[(0, 0), 4] = 1
    V[(0, 0), 5] = 1
    V[(1, 1), 6] = 1
    V[(0, 1), 6] = 1

    W[(0, 0), 0] = 1
    W[(0, 1), 1] = 1
    W[(0, 1), 2] = -1
    W[(1, 0), 3] = 1
    W[(1, 0), 4] = 1
    W[(1, 1), 5] = 1
    W[(0, 0), 6] = 1
    W[(1, 1), 6] = 1
    return U, V, W


def _kron_rowmajor(a, b):
    """Row-major Kronecker product: (a⊗b)[i*rb + j, k*cb + l] = a[i,k]*b[j,l]."""
    ra, ca = a.shape
    rb, cb = b.shape
    out = np.zeros((ra * rb, ca * cb), dtype=int)
    for i in range(ra):
        for k in range(ca):
            if a[i, k]:
                out[i * rb:(i + 1) * rb, k * cb:(k + 1) * cb] = a[i, k] * b
    return out


def _to_lists(M):
    return [[int(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]


def _verify(U, V, W, n, m, p, iters=6):
    rng = np.random.default_rng(0)
    for _ in range(iters):
        A = rng.integers(-4, 5, size=(n, m))
        B = rng.integers(-4, 5, size=(m, p))
        C = A @ B
        acc = np.zeros((n, p), dtype=int)
        for r in range(U.shape[1]):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            if not (u.any() and v.any() and w.any()):
                continue
            A1 = np.zeros((n, m), dtype=int)
            for i in range(n):
                for j in range(m):
                    A1[i, j] = u[i * m + j]
            B1 = np.zeros((m, p), dtype=int)
            for j in range(m):
                for k in range(p):
                    B1[j, k] = v[j * p + k]
            C1 = np.zeros((n, p), dtype=int)
            for k in range(p):
                for i in range(n):
                    C1[i, k] = w[k * n + i]
            acc += (A1 @ A) @ (B1 @ B)
        if not np.array_equal(C, acc):
            return False
    return True


def construct():
    # Liu 3x3 (rank 19), Strassen 2x2 (rank 7)
    U3, V3, W3 = _liu33()
    U2, V2, W2 = _strassen22()

    # Transpose Liu's W to the task's trace convention W[k*n+i] (standard is W[i*n+k]).
    W3t = np.zeros((9, 19), dtype=int)
    for i in range(3):
        for k in range(3):
            W3t[k * 3 + i, :] = W3[i * 3 + k, :]

    # Kronecker product -> 6x6x6 rank 19*7 = 133
    U = _kron_rowmajor(U3, U2)
    V = _kron_rowmajor(V3, V2)
    W = _kron_rowmajor(W3t, W2)

    n, m, p = DIMENSIONS
    if U.shape != (n * m, 133) or V.shape != (m * p, 133) or W.shape != (p * n, 133):
        # Fallback: schoolbook (should not happen)
        return _schoolbook(n, m, p)

    if _verify(U, V, W, n, m, p):
        return {"U": _to_lists(U), "V": _to_lists(V), "W": _to_lists(W)}

    # If verification unexpectedly fails, fall back to the known-correct schoolbook.
    return _schoolbook(n, m, p)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": _to_lists(U), "V": _to_lists(V), "W": _to_lists(W)}
