import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0

    # Schoolbook
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def _verify_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i,j,k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
        return np.allclose(res, 1.0, atol=1e-9)

    # Try to find lower rank using multiple initializations
    for r in range(20, min(45, rank_school)):
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            continue

        found = False
        for trial in range(4):
            if time.time() > deadline - 10:
                break

            # Different initialization strategies
            if trial == 0:
                # Structured: use small integers
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for t in range(r):
                    for idx in range(n*m):
                        U[idx, t] = float(((idx + t) % 3) - 1)
                    for idx in range(m*p):
                        V[idx, t] = float(((idx + t) % 3) - 1)
                    for idx in range(p*n):
                        W[idx, t] = float(((idx + t) % 3) - 1)
            elif trial == 1:
                np.random.seed(42 + r + trial)
                U = np.random.randint(-2, 3, size=(n*m, r)).astype(float) * 0.5
                V = np.random.randint(-2, 3, size=(m*p, r)).astype(float) * 0.5
                W = np.random.randint(-2, 3, size=(p*n, r)).astype(float) * 0.5
            elif trial == 2:
                np.random.seed(123 + r + trial)
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.random.randn(p*n, r)
            else:
                # Start from schoolbook-like structure
                U = U_sb[:, :r].copy()
                V = V_sb[:, :r].copy()
                W = W_sb[:, :r].copy()
                # Add small random perturbation
                U += np.random.randn(n*m, r) * 0.1
                V += np.random.randn(m*p, r) * 0.1
                W += np.random.randn(p*n, r) * 0.1

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def residuals(x):
                U_flat = x[:n*m*r]
                V_flat = x[n*m*r : n*m*r + m*p*r]
                W_flat = x[n*m*r + m*p*r :]

                U_mat = U_flat.reshape(n*m, r)
                V_mat = V_flat.reshape(m*p, r)
                W_mat = W_flat.reshape(p*n, r)

                res = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res[i,j,k] = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])

                return (res - 1).flatten()

            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=600)

                if sol.cost < 1e-8:
                    x_sol = sol.x
                    U_sol = x_sol[:n*m*r].reshape(n*m, r)
                    V_sol = x_sol[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_sol = x_sol[n*m*r + m*p*r :].reshape(p*n, r)

                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2

                    if _verify_decomposition(U_round, V_round, W_round, n, m, p):
                        best_rank = r
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        found = True
                        break
            except Exception:
                pass

        if found:
            break

    def to_list(mat):
        return [[float(val) for val in row] for row in mat]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
