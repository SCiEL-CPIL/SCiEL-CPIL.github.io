"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin for 150s limit

    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    current_rank = rank
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in range(current_rank):
            if time.time() > deadline:
                break
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)

            # Check if decomposition still holds
            T_test = T.copy()
            for r in range(U_test.shape[1]):
                T_test -= np.einsum('i,j,k->ijk', U_test[:,r], V_test[:,r], W_test[:,r])

            if np.allclose(T_test, 0, atol=1e-9):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                best_rank = current_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                improved = True
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
