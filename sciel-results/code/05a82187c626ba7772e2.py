"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 120s limit

    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Try to find a lower rank decomposition
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # Attempt rank 23 using gradient descent
    target_rank = 23
    rng = np.random.default_rng(42)
    U = rng.standard_normal((n * m, target_rank))
    V = rng.standard_normal((m * p, target_rank))
    W = rng.standard_normal((p * n, target_rank))

    lr = 0.01
    for it in range(50000):
        if time.time() > deadline - 15:
            break

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = float(np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :]))
                    e = s - 1.0
                    U[idx_u, :] -= lr * e * V[idx_v, :] * W[idx_w, :]
                    V[idx_v, :] -= lr * e * U[idx_u, :] * W[idx_w, :]
                    W[idx_w, :] -= lr * e * U[idx_u, :] * V[idx_v, :]

        if it % 5000 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                        max_err = max(max_err, abs(s - 1.0))
            if max_err < 1e-8:
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break

    U_out = [[float(x) for x in best_U[i]] for i in range(best_U.shape[0])]
    V_out = [[float(x) for x in best_V[i]] for i in range(best_V.shape[0])]
    W_out = [[float(x) for x in best_W[i]] for i in range(best_W.shape[0])]

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
