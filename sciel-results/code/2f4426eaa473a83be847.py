"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    def als_solve(r, max_iters=2000, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r) * 0.3
        V = rng.randn(m*p, r) * 0.3
        W = np.ones((p*n, r)) * 0.3

        for it in range(max_iters):
            if time.time() > deadline:
                return None

            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = sol
            U = U_new

            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = sol
            V = V_new

            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = sol
            W = W_new

            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(s - 1.0))
            if max_err < 1e-12:
                for scale in [2, 1]:
                    U2 = np.round(U * scale) / scale
                    V2 = np.round(V * scale) / scale
                    W2 = np.round(W * scale) / scale
                    max_err2 = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U2[i*m+j, :], V2[j*p+k, :] * W2[k*n+i, :])
                                max_err2 = max(max_err2, abs(s - 1.0))
                    if max_err2 < 1e-9:
                        return U2, V2, W2
                return None
        return None

    for r in range(28, 85):
        if time.time() > deadline:
            break
        for seed in range(50):
            if time.time() > deadline:
                break
            result = als_solve(r, max_iters=1500, seed=seed)
            if result is not None:
                U, V, W = result
                def clean(x):
                    if abs(x - round(x)) < 1e-9:
                        return int(round(x))
                    elif abs(2*x - round(2*x)) < 1e-9:
                        return round(2*x) / 2
                    return x
                U_list = [[clean(U[i,j]) for j in range(r)] for i in range(n*m)]
                V_list = [[clean(V[i,j]) for j in range(r)] for i in range(m*p)]
                W_list = [[clean(W[i,j]) for j in range(r)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
