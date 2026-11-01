import numpy as np
import time

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s kill

def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]      # 4 x 32
    V = [[0] * rank for _ in range(m * p)]      # 18 x 32
    W = [[0] * rank for _ in range(p * n)]      # 18 x 32

    # Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (U_index, coeff)
    # v_coeffs: list of (V_index_offset, coeff)
    # w_coeffs: list of (W_index_offset, coeff)
    strassen_terms = [
        # P0 = (A00 + A11) * (B00 + B11)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1), (1, -1), (2, 1) ]),
        # P1 = (A10 + A11) * B00
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        # P2 = A00 * (B01 - B11)
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        # P3 = A11 * (B10 - B00)
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, -1) ]),
        # P4 = (A00 + A01) * B11
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (3, 1) ]),
        # P5 = (A10 - A00) * (B00 + B01)
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (0, -1), (2, 1) ]),
        # P6 = (A01 - A11) * (B10 + B11)
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (1, 1) ]),
    ]

    # Fill in 4 Strassen blocks for columns 0-7
    for block in range(4):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]

            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff

            for v_idx_off, coeff in v_coeffs:
                v_idx = 2 * block + v_idx_off
                V[v_idx][col] += coeff

            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off
                W[w_idx][col] += coeff

    # Schoolbook terms for column 8 (2x2x1 block)
    # V indices: V[j*p + k], so for k=8: V[8] (j=0), V[17] (j=1)
    # W indices: W[k*n + i], so for k=8: W[16] (i=0), W[17] (i=1)
    schoolbook_terms = [
        ([ (0, 1) ], [ (8, 1) ],  [ (16, 1) ]),   # A00 * B0 -> C00
        ([ (1, 1) ], [ (17, 1) ], [ (16, 1) ]),   # A01 * B1 -> C00
        ([ (2, 1) ], [ (8, 1) ],  [ (17, 1) ]),   # A10 * B0 -> C10
        ([ (3, 1) ], [ (17, 1) ], [ (17, 1) ]),   # A11 * B1 -> C10
    ]

    col_start = 4 * 7  # column 28
    for t in range(4):
        col = col_start + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]

        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff

        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff

        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff

    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)

def _try_rank_31(n, m, p):
    """
    Try to find a rank-31 decomposition by checking if any single term
    in the rank-32 solution is redundant.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
        return None

    U32, V32, W32 = _build_rank_32(n, m, p)
    Ua = np.array(U32, dtype=float)
    Va = np.array(V32, dtype=float)
    Wa = np.array(W32, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    for r in range(32):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        mask = [i for i in range(32) if i != r]
        U = Ua[:, mask]
        V = Va[:, mask]
        W = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T, T_recon, atol=1e-10):
            return (
                [[U[i][j] for j in range(31)] for i in range(n * m)],
                [[V[i][j] for j in range(31)] for i in range(m * p)],
                [[W[i][j] for j in range(31)] for i in range(p * n)],
            )
    return None

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME

    # Try rank 31 first if we have enough time
    if elapsed < _TIMEOUT - 15:
        result = _try_rank_31(n, m, p)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Fall back to rank-32 Strassen-based decomposition
    U, V, W = _build_rank_32(n, m, p)
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Final fallback: schoolbook rank 36
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
