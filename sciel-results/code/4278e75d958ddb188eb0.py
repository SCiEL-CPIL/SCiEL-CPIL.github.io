"""Matrix multiplication 4x4x4 tensor decomposition using recursive Strassen (rank 49)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Inner 2x2 Strassen pattern (rank 7).
# Each entry: u = left factor terms, v = right factor terms, w = output terms.
# Each term is (row, col, sign).
# ---------------------------------------------------------------------------
_INNER_STRASSEN_2X2 = [
    {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
    {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)],          'w': [(1,0,1),(1,1,-1)]},
    {'u': [(0,0,1)],          'v': [(0,1,1),(1,1,-1)],'w': [(0,1,1),(1,1,1)]},
    {'u': [(1,1,1)],          'v': [(1,0,1),(0,0,-1)],'w': [(0,0,1),(1,0,1)]},
    {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)],          'w': [(0,1,1),(0,0,-1)]},
    {'u': [(1,0,1),(0,0,-1)],'v': [(0,0,1),(0,1,1)],  'w': [(1,1,1)]},
    {'u': [(0,1,1),(1,1,-1)],'v': [(1,0,1),(1,1,1)],  'w': [(0,0,1)]},
]

# ---------------------------------------------------------------------------
# Outer 4x4 Strassen pattern (rank 7) over 2x2 blocks.
# a = A block terms, b = B block terms, c = C block terms.
# ---------------------------------------------------------------------------
_OUTER_STRASSEN_4X4 = [
    {'a': [(0,0,1),(1,1,1)], 'b': [(0,0,1),(1,1,1)], 'c': [(0,0,1),(1,1,1)]},
    {'a': [(1,0,1),(1,1,1)], 'b': [(0,0,1)],          'c': [(1,0,1),(1,1,-1)]},
    {'a': [(0,0,1)],          'b': [(0,1,1),(1,1,-1)],'c': [(0,1,1),(1,1,1)]},
    {'a': [(1,1,1)],          'b': [(1,0,1),(0,0,-1)],'c': [(0,0,1),(1,0,1)]},
    {'a': [(0,0,1),(0,1,1)], 'b': [(1,1,1)],          'c': [(0,1,1),(0,0,-1)]},
    {'a': [(1,0,1),(0,0,-1)],'b': [(0,0,1),(0,1,1)],  'c': [(1,1,1)]},
    {'a': [(0,1,1),(1,1,-1)],'b': [(1,0,1),(1,1,1)],  'c': [(0,0,1)]},
]


def _build_strassen_4x4(n, m, p):
    """Build rank-49 U, V, W for 4x4x4 via recursive Strassen."""
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for s, outer in enumerate(_OUTER_STRASSEN_4X4):
        for t, inner in enumerate(_INNER_STRASSEN_2X2):
            r = s * 7 + t

            # U: A[row, col] -> index row*m + col
            for (P, Q, sa) in outer['a']:
                for (ir, jc, sx) in inner['u']:
                    coeff = sa * sx
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        U[row * m + col, r] += coeff

            # V: B[row, col] -> index row*p + col
            for (P, Q, sb) in outer['b']:
                for (ir, jc, sy) in inner['v']:
                    coeff = sb * sy
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        V[row * p + col, r] += coeff

            # W: C[row, col] -> W index = col*n + row (trace convention)
            for (P, Q, sc) in outer['c']:
                for (ir, jc, sz) in inner['w']:
                    coeff = sc * sz
                    if coeff:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        W[col * n + row, r] += coeff

    return U, V, W


def _verify(U, V, W, n, m, p, atol=1e-9):
    """Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] == 1."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ikr,jkr,jir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))


def _schoolbook(n, m, p):
    """Fallback rank n*m*p decomposition."""
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
    t0 = time.time()
    deadline = t0 + 115.0
    n = m = p = 4

    # Primary: recursive Strassen, rank 49 (optimal)
    try:
        U, V, W = _build_strassen_4x4(n, m, p)
        if time.time() < deadline and _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook rank 64
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
