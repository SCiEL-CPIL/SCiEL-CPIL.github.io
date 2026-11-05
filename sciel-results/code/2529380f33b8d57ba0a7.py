import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 130

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True

    def als_decompose(R, seed=42, max_iters=3000):
        if time.time() > deadline - 10:
            return None

        rng = np.random.RandomState(seed)

        U = rng.randn(n*m, R)
        V = rng.randn(m*p, R)
        W = rng.randn(p*n, R)

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, R))
                    for j in range(m):
                        B[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass

            for i in range(n):
                for j in range(m):
                    C = np.zeros((p, R))
                    for k in range(p):
                        C[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(C, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass

            for j in range(m):
                for k in range(p):
                    D = np.zeros((n, R))
                    for i in range(n):
                        D[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(D, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass

            if it % 500 == 0:
                max_err = max(abs(np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1)
                             for i in range(n) for j in range(m) for k in range(p))
                if max_err < 1e-10:
                    return U.tolist(), V.tolist(), W.tolist()

        max_err = max(abs(np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1)
                     for i in range(n) for j in range(m) for k in range(p))
        if max_err < 1e-10:
            return U.tolist(), V.tolist(), W.tolist()
        return None

    for R in range(55, 64):
        if time.time() > deadline - 5:
            break
        for seed in range(30):
            if time.time() > deadline - 5:
                break
            result = als_decompose(R, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                if verify(U_list, V_list, W_list):
                    best_rank = R
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        if best_rank < 63:
            break

    return best_result
