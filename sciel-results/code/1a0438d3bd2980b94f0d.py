"""Matrix multiplication 4x4x4 tensor decomposition using recursive Strassen (rank 49)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Inner 2x2 Strassen pattern (rank 7).
# Each entry: u = [(row, col, sign)], v = [(row, col, sign)], w = [(row, col, sign)]
# ---------------------------------------------------------------------------
_INNER_STRASSEN_2X2 = [
    # t=0: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
    {'u': [(0, 0, 1), (1, 1, 1)], 'v': [(0, 0, 1), (1, 1, 1)], 'w': [(0, 0, 1), (1, 1, 1)]},
    # t=1: (X10+X11)Y00       -> Z10 += 1, Z11 -= 1
    {'u': [(1, 0, 1), (1, 1, 1)], 'v': [(0, 0, 1)], 'w': [(1, 0, 1), (1, 1, -1)]},
    # t=2: X00(Y01-Y11)       -> Z01 += 1, Z11 += 1
    {'u': [(0, 0, 1)], 'v': [(0, 1, 1), (1, 1, -1)], 'w': [(0, 1, 1), (1, 1, 1)]},
    # t=3: X11(Y10-Y00)       -> Z00 += 1, Z10 += 1
    {'u': [(1, 1, 1)], 'v': [(1, 0, 1), (0, 0, -1)], 'w': [(0, 0, 1), (1, 0, 1)]},
    # t=4: (X00+X01)Y11       -> Z01 += 1, Z00 -= 1
    {'u': [(0, 0, 1), (0, 1, 1)], 'v': [(1, 1, 1)], 'w': [(0, 1, 1), (0, 0, -1)]},
    # t=5: (X10-X00)(Y00+Y01) -> Z11 += 1
    {'u': [(1, 0, 1), (0, 0, -1)], 'v': [(0, 0, 1), (0, 1, 1)], 'w': [(1, 1, 1)]},
    # t=6: (X01-X11)(Y10+Y11) -> Z00 += 1
    {'u': [(0, 1, 1), (1, 1, -1)], 'v': [(1, 0, 1), (1, 1, 1)], 'w': [(0, 0, 1)]},
]

# ---------------------------------------------------------------------------
# Outer 4x4 Strassen pattern (rank 7) over 2x2 blocks.
#   a: which A blocks (block_row, block_col, sign)
#   b: which B blocks (block_row, block_col, sign)
#   c: which C blocks (block_row, block_col, sign)
# ---------------------------------------------------------------------------
_OUTER_STRASSEN_4X4 = [
    # s=0: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    {'a': [(0, 0, 1), (1, 1, 1)], 'b': [(0, 0, 1), (1, 1, 1)], 'c': [(0, 0, 1), (1, 1, 1)]},
    # s=1: (A10+A11)B00       -> C10 += 1, C11 -= 1
    {'a': [(1, 0, 1), (1, 1, 1)], 'b': [(0, 0, 1)], 'c': [(1, 0, 1), (1, 1, -1)]},
    # s=2: A00(B01-B11)       -> C01 += 1, C11 += 1
    {'a': [(0, 0, 1)], 'b': [(0, 1, 1), (1, 1, -1)], 'c': [(0, 1, 1), (1, 1, 1)]},
    # s=3: A11(B10-B00)       -> C00 += 1, C10 += 1
    {'a': [(1, 1, 1)], 'b': [(1, 0, 1), (0, 0, -1)], 'c': [(0, 0, 1), (1, 0, 1)]},
    # s=4: (A00+A01)B11       -> C01 += 1, C00 -= 1
    {'a': [(0, 0, 1), (0, 1, 1)], 'b': [(1, 1, 1)], 'c': [(0, 1, 1), (0, 0, -1)]},
    # s=5: (A10-A00)(B00+B01) -> C11 += 1
    {'a': [(1, 0, 1), (0, 0, -1)], 'b': [(0, 0, 1), (0, 1, 1)], 'c': [(1, 1, 1)]},
    # s=6: (A01-A11)(B10+B11) -> C00 += 1
    {'a': [(0, 1, 1), (1, 1, -1)], 'b': [(1, 0, 1), (1, 1, 1)], 'c': [(0, 0, 1)]},
]


def _strassen_4x4(n, m, p):
    """
    Build the rank-49 U, V, W factor matrices for 4x4 x 4x4 multiplication by
    composing the outer Strassen (block level) with the inner Strassen (2x2 level).
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U index : i*m + k   (A row i, A col k)
      V index : k*p + j   (B row k, B col j)
      W index : j*n + i   (C col j, C row i)  -- official trace convention
    """
    assert (n, m, p) == (4, 4, 4)
    rank = len(_OUTER_STRASSEN_4X4) * len(_INNER_STRASSEN_2X2)  # 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for s, outer in enumerate(_OUTER_STRASSEN_4X4):
        for t, inner in enumerate(_INNER_STRASSEN_2X2):
            r = s * len(_INNER_STRASSEN_2X2) + t

            # U: A block (P, Q) holds elements at A[2P+ir, 2Q+jc]
            for (P, Q, sa) in outer['a']:
                for (ir, jc, sx) in inner['u']:
                    coeff = sa * sx
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        U[row * m + col, r] += coeff

            # V: B block (P, Q) holds elements at B[2P+ir, 2Q+jc]
            for (P, Q, sb) in outer['b']:
                for (ir, jc, sy) in inner['v']:
                    coeff = sb * sy
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        V[row * p + col, r] += coeff

            # W: C block (P, Q) holds elements at C[2P+ir, 2Q+jc]
            #    W index = col * n + row  (== j*n + i)
            for (P, Q, sc) in outer['c']:
                for (ir, jc, sz) in inner['w']:
                    coeff = sc * sz
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        W[col * n + row, r] += coeff

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify the decomposition against the matrix multiplication tensor using the
    official trace convention.
    T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1
    """
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)

    # T[i,j,k] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    T = np.einsum('ikr,kjr,jir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))


def _schoolbook_decomposition(n, m, p):
    """Fallback rank n*m*p decomposition (always valid)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1.0
                V[k * p + j, col] = 1.0
                W[j * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    Returns dict with keys 'U', 'V', 'W' as lists.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 150s kill

    n = m = p = 4

    # Primary: recursive Strassen (rank 49)
    try:
        U, V, W = _strassen_4x4(n, m, p)
        if time.time() < deadline and _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook (rank 64)
    U, V, W = _schoolbook_decomposition(n, m, p)
    if time.time() < deadline:
        _verify_decomposition(U, V, W, n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
