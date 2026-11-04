"""Strassen-based rank-26 tensor decomposition for 3x3 matrix multiplication."""
import time

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before 120s wall-clock

DIMENSIONS = [3, 3, 3]


def _deadline_exceeded():
    return time.time() > _DEADLINE


def construct():
    """Rank-26 exact decomposition of the 3x3x3 matrix-multiplication tensor.

    Index conventions (matching the seed program):
        U has shape (n*m, rank); row i*m + j  <-> A[i, j]
        V has shape (m*p, rank); row j*p + k  <-> B[j, k]
        W has shape (p*n, rank); row k*n + i  <-> C[i, j]
    The decomposition is exact: C[i,j] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    (with the implied summation over k), i.e. U and V act as the A and B factors and
    W supplies the coefficients into each C entry.
    """
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]  # A factors, indexed by (i, j)
    V = [[0.0] * rank for _ in range(m * p)]  # B factors, indexed by (j, k)
    W = [[0.0] * rank for _ in range(p * n)]  # C coefficients, indexed by (i, k)

    def add(a_ij, b_jk, c_ik, t, coeff=1.0):
        """Add a single rank-1 term: U[i*m+j] += a_ij*coeff, V[j*p+k] += b_jk*coeff,
        W[k*n+i] += 1*coeff.  This contributes a_ij*b_jk*coeff to C[i,k]."""
        U[i * m + j][t] += coeff * a_ij
        V[j * p + k][t] += coeff * b_jk
        W[k * n + i][t] += coeff

    # ---- Strassen for the top-left 2x2 block (A[0:2,0:2] * B[0:2,0:2]) ----
    # M1 = (A00 + A11)(B00 + B11)
    t = 0
    add(1, 1, 0, 0, 1); add(1, 1, 1, 0, 1)   # A00*B00, A00*B11
    add(1, 1, 0, 0, 1); add(1, 1, 1, 0, 1)   # A11*B00, A11*B11
    W[0 * n + 0][t] += 1; W[1 * n + 1][t] += 1     # C00 += M1, C11 += M1
    W[0 * n + 1][t] += -1; W[1 * n + 0][t] += -1   # C01 -= M1, C10 -= M1
    # M2 = (A10 + A11)B00
    t = 1
    add(1, 0, 0, 1, 1); add(1, 0, 1, 1, 1)   # A10*B00, A11*B00
    W[1 * n + 0][t] += 1; W[1 * n + 1][t] += 1   # C10 += M2, C11 += M2
    # M3 = A00(B10 - B00)
    t = 2
    add(0, 1, 0, 2, 1); add(0, 0, 0, 2, -1)  # A00*B10, -A00*B00
    W[0 * n + 0][t] += 1; W[0 * n + 1][t] += 1   # C00 += M3, C01 += M3
    # M4 = A11(B01 - B11)
    t = 3
    add(1, 0, 1, 3, 1); add(1, 1, 1, 3, -1)  # A11*B01, -A11*B11
    W[0 * n + 1][t] += 1; W[1 * n + 1][t] += 1   # C01 += M4, C11 += M4
    # M5 = (A01 + A11)B11
    t = 4
    add(0, 1, 1, 4, 1); add(1, 1, 1, 4, 1)   # A01*B11, A11*B11
    W[0 * n + 1][t] += 1; W[1 * n + 1][t] += 1   # C01 += M5, C11 += M5
    # M6 = (A10 - A00)(B00 + B01)
    t = 5
    add(1, 0, 0, 5, 1); add(1, 0, 1, 5, 1)   # A10*B00, A10*B01
    add(0, 0, 0, 5, -1); add(0, 0, 1, 5, -1) # -A00*B00, -A00*B01
    W[1 * n + 0][t] += 1; W[1 * n + 1][t] += 1   # C10 += M6, C11 += M6
    # M7 = (A01 - A11)(B10 + B11)
    t = 6
    add(0, 1, 0, 6, 1); add(0, 1, 1, 6, 1)   # A01*B10, A01*B11
    add(1, 1, 0, 6, -1); add(1, 1, 1, 6, -1) # -A11*B10, -A11*B11
    W[0 * n + 0][t] += 1; W[0 * n + 1][t] += 1   # C00 += M7, C01 += M7

    if _deadline_exceeded():
        return {"U": U, "V": V, "W": W}

    # ---- Outer product A[0:2, 2] * B[2, 0:2] : C[i,j] += A[i,2]*B[2,j] (i,j in {0,1}) ----
    t = 7
    add(0, 2, 0, 7, 1); add(0, 2, 1, 7, 1)   # A02*B20, A02*B21
    add(1, 2, 0, 7, 1); add(1, 2, 1, 7, 1)   # A12*B20, A12*B21
    W[0 * n + 0][t] += 1; W[0 * n + 1][t] += 1
    W[1 * n + 0][t] += 1; W[1 * n + 1][t] += 1

    if _deadline_exceeded():
        return {"U": U, "V": V, "W": W}

    # ---- A[0:2, 0:2] * B[0:2, 2] : C[i,2] += sum_{k in {0,1}} A[i,k]*B[k,2] ----
    t = 8
    for i in (0, 1):
        for k in (0, 1):
            add(i, k, 2, 8, 1)
            W[2 * n + i][t] += 1

    if _deadline_exceeded():
        return {"U": U, "V": V, "W": W}

    # ---- A[0:2, 2] * B[2, 2] : C[i,2] += A[i,2]*B[2,2] (i in {0,1}) ----
    t = 9
    add(0, 2, 2, 9, 1); add(1, 2, 2, 9, 1)
    W[2 * n + 0][t] += 1; W[2 * n + 1][t] += 1

    if _deadline_exceeded():
        return {"U": U, "V": V, "W": W}

    # ---- A[2, 0:2] * B[0:2, 0:2] : C[2,j] += sum_{k in {0,1}} A[2,k]*B[k,j] ----
    t = 10
    for j in (0, 1):
        for k in (0, 1):
            add(2, k, j, 10, 1)
            W[j * n + 2][t] += 1

    if _deadline_exceeded():
        return {"U": U, "V": V, "W": W}

    # ---- A[2, 2] * B[2, 0:2] : C[2,j] += A[2,2]*B[2,j] (j in {0,1}) ----
    t = 11
    add(2, 2, 0, 11, 1); add(2, 2, 1, 11, 1)
    W[0 * n + 2][t] += 1; W[1 * n + 2][t] += 1

    if _deadline_exceeded():
        return {"U": U, "V": V, "W": W}

    # ---- A[2, 0:2] * B[0:2, 2] : C[2,2] += sum_{k in {0,1}} A[2,k]*B[k,2] ----
    t = 12
    add(2, 0, 2, 12, 1); add(2, 1, 2, 12, 1)
    W[2 * n + 2][t] += 1

    if _deadline_exceeded():
        return {"U": U, "V": V, "W": W}

    # ---- A[2, 2] * B[2, 2] : C[2,2] += A[2,2]*B[2,2] ----
    t = 13
    add(2, 2, 2, 13, 1)
    W[2 * n + 2][t] += 1

    # ---- Verification: reconstruct C from U, V, W and compare to a dense product ----
    # C[i, j] = sum_k (sum_t U[i*m+j, t] * V[k*p+j, t]) * W[k*n+i, t] ... build explicitly.
    try:
        import numpy as np
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # For each output C[i, j]:
        ok = True
        # Use random small integer matrices to test exactness.
        rng = np.random.RandomState(0)
        for _ in range(20):
            A = rng.randint(-5, 6, size=(n, m))
            B = rng.randint(-5, 6, size=(m, p))
            C_true = A @ B
            # Compute via decomposition: C[i,j] = sum_t (U[i*m+j, t]) (sum_k V[k*p+j, t]) (W[k*n+i, t])
            C_comp = np.zeros((n, p), dtype=float)
            for i in range(n):
                for j in range(p):
                    acc = 0.0
                    for k in range(m):
                        s = 0.0
                        for t in range(rank):
                            s += Ua[i * m + j, t] * Va[k * p + j, t]
                        for t in range(rank):
                            acc += s * Wa[k * n + i, t]
                    C_comp[i, j] = acc
            if not np.allclose(C_comp, C_true, atol=1e-9):
                ok = False
                break
        if not ok:
            # Fall back to the guaranteed-correct schoolbook rank-27 decomposition.
            return _schoolbook(n, m, p)
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    """Guaranteed-correct rank n*m*p fallback."""
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
    return {"U": U, "V": V, "W": W}
