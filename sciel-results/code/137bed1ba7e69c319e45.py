import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1

    def verify(U, V, W, num_tests=5):
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(U.shape[1]):
                a_proj = np.dot(U[:, r], A.flatten())
                b_proj = np.dot(V[:, r], B.flatten())
                for k in range(p):
                    for i in range(n):
                        C_computed[i, k] += W[k * n + i, r] * a_proj * b_proj
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    np.random.seed(789)

    for R in range(55, 44, -1):
        if time.time() > deadline - 10:
            break

        for attempt in range(30):
            if time.time() > deadline - 10:
                break

            U = np.random.randn(n * m, R) * 0.3
            V = np.random.randn(m * p, R) * 0.3
            W = np.random.randn(p * n, R) * 0.3

            for it in range(100):
                if time.time() > deadline - 10:
                    break

                # Update U
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, R))
                        for k in range(p):
                            A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                        U[i * m + j, :] = x

                # Update V
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, R))
                        for i in range(n):
                            A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                        V[j * p + k, :] = x

                # Update W
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, R))
                        for j in range(m):
                            A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        W[k * n + i, :] = x

            if verify(U, V, W):
                best_rank = R
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
