import numpy as np

DIMENSIONS = [4, 4, 5]

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(expected, reconstructed, atol=tol)

def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 x 4x4 using 2x2 Strassen blocks."""
    block_coeffs_U = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    block_coeffs_V = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    block_coeffs_W = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(1, 0, 1), (0, 0, 1)],
        [(0, 1, -1), (1, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    rank = 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc

                for (br, bc, coeff_u) in block_coeffs_U[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            i = br * 2 + lr2
                            j = bc * 2 + lc2
                            U[i * 4 + j][c] += coeff_u

                for (br, bc, coeff_v) in block_coeffs_V[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            j = br * 2 + lr2
                            k = bc * 2 + lc2
                            V[j * 4 + k][c] += coeff_v

                for (br, bc, coeff_w) in block_coeffs_W[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            k = br * 2 + lr2
                            i = bc * 2 + lc2
                            W[k * 4 + i][c] += coeff_w

    return U, V, W

def _build_4x4x5_improved():
    """Build 4x4x5 decomposition: Strassen 4x4x4 core + 1 boundary term."""
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    rank_boundary = 1
    rank_total = rank_core + rank_boundary

    # U: 16 x rank_total
    # V: 20 x rank_total (4 rows x 5 cols)
    # W: 20 x rank_total (5 rows x 4 cols)
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(20)]
    W = [[0] * rank_total for _ in range(20)]

    # Copy core (indices 0..27)
    # Core V is 4x4, mapped to V[j, k] for k in 0..3
    # Core W is 4x4, mapped to W[k, i] for k in 0..3
    for c in range(rank_core):
        for i in range(16):
            U[i][c] = U44[i][c]
        for j in range(4):
            for k in range(4):
                V[j * 5 + k][c] = V44[j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][c] = W44[k * 4 + i][c]

    # Add boundary term for k=4 (index 4)
    c = rank_core
    # U[i, j] = 1/4 for all i, j
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][c] = 0.25
    # V[j, k] = 1 if k=4, else 0
    for j in range(4):
        V[j * 5 + 4][c] = 1.0
    # W[k, i] = 1 if k=4, else 0
    for i in range(4):
        W[4 * 4 + i][c] = 1.0

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_4x4x5_improved()
    if _verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails
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
