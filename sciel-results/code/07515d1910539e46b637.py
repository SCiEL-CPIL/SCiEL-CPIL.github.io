"""Matrix multiplication 3,3,5 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_a, dim_b, dim_c = n * m, m * p, p * n  # 9, 15, 15

    # Build dense tensor
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    deadline = time.time() + 105
    best = None

    # Precompute index arrays for speed
    ab_idx = np.array([(a, b) for a in range(dim_a) for b in range(dim_b)])
    ac_idx = np.array([(a, c) for a in range(dim_a) for c in range(dim_c)])
    bc_idx = np.array([(b, c) for b in range(dim_b) for c in range(dim_c)])

    for rank in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40, 42, 44, 45]:
        if time.time() > deadline - 6:
            break
        for trial in range(8):
            if time.time() > deadline - 2:
                break
            np.random.seed(rank * 1000 + trial * 13)
            U = np.random.randn(dim_a, rank)
            V = np.random.randn(dim_b, rank)
            W = np.random.randn(dim_c, rank)

            for it in range(35):
                if time.time() > deadline - 1:
                    break

                # ALS updates
                for a in range(dim_a):
                    A = (V[:, :, None] * W[None, :, :]).reshape(dim_b * dim_c, rank)
                    U[a], _, _, _ = np.linalg.lstsq(A, T[a].ravel(), rcond=None)
                for b in range(dim_b):
                    A = (U[:, :, None] * W[None, :, :]).reshape(dim_a * dim_c, rank)
                    V[b], _, _, _ = np.linalg.lstsq(A, T[:, b, :].ravel(), rcond=None)
                for c in range(dim_c):
                    A = (U[:, :, None] * V[None, :, :]).reshape(dim_a * dim_b, rank)
                    W[c], _, _, _ = np.linalg.lstsq(A, T[:, :, c].ravel(), rcond=None)

                if it % 4 == 0:
                    approx = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(approx - T))
                    if err < 1e-2:
                        best = (rank, U, V, W)
                        break

            if best:
                break
        if best:
            break

    if best:
        r, U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Schoolbook fallback
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
