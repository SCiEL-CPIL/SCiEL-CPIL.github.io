import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 105.0

    # Build the target tensor T[a, b, d] where a=i*m+j, b=j*p+k, d=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0

    schoolbook_rank = n * m * p

    def check_decomposition(U, V, W, rank):
        recon = np.zeros((n * m, m * p, p * n))
        for c in range(rank):
            recon += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
        return np.max(np.abs(recon - T)) < 1e-7

    best_result = None
    # Try ranks from 30 up to schoolbook
    for rank in range(30, schoolbook_rank + 1):
        if time.time() > DEADLINE:
            break
        for trial in range(40):
            if time.time() > DEADLINE:
                break
            np.random.seed(trial + rank * 1000)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)

            for it in range(100):
                # Update U
                M = (V * W).reshape(m * p * p * n, rank)
                T_a = T.reshape(n * m, -1)
                try:
                    U = np.linalg.lstsq(M.T, T_a.T, rcond=None)[0].T
                except Exception:
                    break

                # Update V
                M = (U * W).reshape(n * m * p * n, rank)
                T_b = np.transpose(T, (1, 0, 2)).reshape(m * p, -1)
                try:
                    V = np.linalg.lstsq(M.T, T_b.T, rcond=None)[0].T
                except Exception:
                    break

                # Update W
                M = (U * V).reshape(n * m * m * p, rank)
                T_d = np.transpose(T, (2, 0, 1)).reshape(p * n, -1)
                try:
                    W = np.linalg.lstsq(M.T, T_d.T, rcond=None)[0].T
                except Exception:
                    break

                if check_decomposition(U, V, W, rank):
                    best_result = (U.copy(), V.copy(), W.copy(), rank)
                    break

            if best_result:
                break
        if best_result:
            break

    if best_result:
        U, V, W, rank = best_result
        U_list = [[float(U[i][c]) for c in range(rank)] for i in range(n * m)]
        V_list = [[float(V[i][c]) for c in range(rank)] for i in range(m * p)]
        W_list = [[float(W[i][c]) for c in range(rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook decomposition
    rank = schoolbook_rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
