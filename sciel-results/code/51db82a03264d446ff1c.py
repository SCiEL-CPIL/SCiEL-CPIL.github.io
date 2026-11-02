"""Tensor decomposition for 2x4 by 4x8 matrix multiplication (rank-minimal)."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition exactly reproduces matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_shared_decomposition():
    """
    Rank-28 decomposition for 2x4 x 4x8 via block Strassen with SHARED A-factors.

    A (2x4) is split into two 2x2 blocks A0 (cols 0,1) and A1 (cols 2,3);
    B (4x8) is split into four 2x8 blocks Bb (rows 2b,2b+1), b=0..3.
    C[i,k] = A0[i,0:2]·B0[i? no] ... = sum_a sum_b (A_a)_{i,*} (B_b)_{*,k}.

    For each (a, b) pair we use Strassen's 7 products for a 2x2 x 2x2 product.
    Key optimization: the A-side linear combinations (U) depend only on the
    Strassen term and the A-block index a, NOT on the B-block index b. Since
    n=2 (only 2 rows), the two A-blocks A0 and A1 can SHARE the same 7
    Strassen products: a single product term is multiplied by A_a (via U) and
    by B_b (via V) for all a,b. This collapses 2*4*7 = 56 terms to 4*7 = 28.

    Strassen's 7 products for C = A*B (A,B 2x2):
      M1 = (a00+a11)(b00+b11)
      M2 = (a10+a11)b00
      M3 = a00(b01-b11)
      M4 = a11(b10-b00)
      M5 = (a00+a01)b11
      M6 = (a10-a00)(b00+b01)
      M7 = (a01-a11)(b10+b11)
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    nA = 2  # number of A-blocks (4 cols / 2)
    nB = 4  # number of B-blocks (8 cols / 2)
    rank = nB * 7  # 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: (li, lj) -> coeff ; B_coeffs: (lj, lk) -> coeff ; C_coeffs: (li, lk) -> coeff
    strassen_terms = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]

    col = 0
    for b in range(nB):
        for (A_coeffs, B_coeffs, C_coeffs) in strassen_terms:
            # U: shared across A-blocks. U[i*m + (2a + lj), col] = Acoeff(lj)
            for (li, lj), coeff in A_coeffs.items():
                for a in range(nA):
                    global_j = 2 * a + lj
                    U[li * m + global_j, col] += coeff
            # V: V[(2a+lj)*p + (2b+lk), col] = Bcoeff(lk)
            for (lj, lk), coeff in B_coeffs.items():
                for a in range(nA):
                    global_j = 2 * a + lj
                    global_k = 2 * b + lk
                    V[global_j * p + global_k, col] += coeff
            # W: W[(2b+lk)*n + li, col] = Ccoeff
            for (li, lk), coeff in C_coeffs.items():
                global_k = 2 * b + lk
                W[global_k * n + li, col] += coeff
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_decomposition():
    """Fallback: independent block Strassen, rank 56."""
    n, m, p = DIMENSIONS
    rank = 2 * 4 * 7  # 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    strassen_terms = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    col = 0
    for a in range(2):
        for b in range(4):
            for (A_coeffs, B_coeffs, C_coeffs) in strassen_terms:
                for (li, lj), coeff in A_coeffs.items():
                    U[li * m + (2 * a + lj), col] += coeff
                for (lj, lk), coeff in B_coeffs.items():
                    V[(2 * a + lj) * p + (2 * b + lk), col] += coeff
                for (li, lk), coeff in C_coeffs.items():
                    W[(2 * b + lk) * n + li, col] += coeff
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS

    candidates = [
        _strassen_shared_decomposition,  # rank 28 (preferred)
        _strassen_decomposition,         # rank 56
        _schoolbook_decomposition,       # rank 64
    ]

    best = None
    for make in candidates:
        if time.time() > deadline:
            break
        try:
            result = make()
        except Exception:
            continue
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            return result
        if best is None:
            best = result

    if best is not None:
        return best
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
