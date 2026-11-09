import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    rank_sb = n * m * p

    U0 = np.zeros((n*m, rank_sb))
    V0 = np.zeros((m*p, rank_sb))
    W0 = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1.0
                V0[j*p+k, col] = 1.0
                W0[k*n+i, col] = 1.0
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U0, V0, W0

    def validate(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def als_decompose(rank, max_iter=400, seed=0):
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, rank) * 0.05
        V = rng.randn(m*p, rank) * 0.05
        W = rng.randn(p*n, rank) * 0.05

        prev_err = float('inf')
        for it in range(max_iter):
            for i in range(n):
                for j in range(m):
                    w_idx = np.array([k*n + i for k in range(p)])
                    M = np.outer(V[j*p:(j+1)*p, :], W[w_idx, :].T)
                    U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]

            for j in range(m):
                for k in range(p):
                    u_idx = np.array([i*m + j for i in range(n)])
                    w_idx = np.array([k*n + i for i in range(n)])
                    M = np.outer(U[u_idx, :], W[w_idx, :].T)
                    V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]

            for k in range(p):
                for i in range(n):
                    u_idx = np.array([i*m + j for j in range(m)])
                    v_idx = np.array([j*p + k for j in range(m)])
                    M = np.outer(U[u_idx, :], V[v_idx, :].T)
                    W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]

            if it % 25 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err

        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return U, V, W, err

    for rank in range(rank_sb - 1, max(rank_sb - 50, 0), -1):
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(8):
            if time.time() > deadline - 10:
                break
            U, V, W, err = als_decompose(rank, seed=seed)
            if err < 1e-10:
                if validate(U, V, W):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        if found:
            break

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    return {"U": U_list, "V": V_list, "W": W_list}
