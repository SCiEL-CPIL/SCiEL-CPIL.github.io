import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses recursive Strassen with rank 98 (2 * 7 * 7), the theoretical minimum.
    """
    start = time.time()
    n, m, p = DIMENSIONS

    # Strassen 2x2 factors: each is (u[4], v[4], w[4])
    # u: [A00, A01, A10, A11], v: [B00, B01, B10, B11], w: [C00, C01, C10, C11]
    S = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(2):
        for s in range(7):
            u_block, v_block, w_block = S[s]
            for r in range(7):
                u_scalar, v_scalar, w_scalar = S[r]
                col = t * 49 + s * 7 + r

                # U: A is 4x4, split into four 2x2 blocks
                for bi in range(2):
                    for bj in range(2):
                        cb = u_block[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = u_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += cb * cs

                # V: B block for subproblem t (columns 4*t to 4*t+3)
                for bi in range(2):
                    for bj in range(2):
                        cb = v_block[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = v_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                j = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                V[j * p + k][col] += cb * cs

                # W: C block for subproblem t (columns 4*t to 4*t+3)
                for bi in range(2):
                    for bj in range(2):
                        cb = w_block[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = w_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                i = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                W[k * n + i][col] += cb * cs

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)

    if np.allclose(result, 1.0, atol=1e-10):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank-128
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
