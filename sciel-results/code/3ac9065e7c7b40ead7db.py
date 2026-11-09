import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 130  # Leave 20s margin for safety

    # Generate valid triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                triples.append((u, v, w))

    num_triples = len(triples)
    u_indices = np.array([t[0] for t in triples])
    v_indices = np.array([t[1] for t in triples])
    w_indices = np.array([t[2] for t in triples])

    def verify_solution(U, V, W, atol=1e-9):
        for idx in range(num_triples):
            u, v, w = triples[idx]
            val = np.sum(U[u, :] * V[v, :] * W[w, :])
            if not np.isclose(val, 1.0, atol=atol):
                return False
        return True

    def polish_solution(x, target_rank, size_u, size_v, size_w):
        U = x[:size_u].reshape(n*m, target_rank)
        V = x[size_u:size_u+size_v].reshape(m*p, target_rank)
        W = x[size_u+size_v:].reshape(p*n, target_rank)

        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round

        for offset in [0.25, -0.25, 0.5, -0.5, 0.125, -0.125]:
            U_r = np.round((U + offset) * 2) / 2
            V_r = np.round((V + offset) * 2) / 2
            W_r = np.round((W + offset) * 2) / 2
            if verify_solution(U_r, V_r, W_r):
                return U_r, V_r, W_r

        return None

    def try_rank(target_rank, max_attempts=35):
        elapsed = time.time() - start_time
        if elapsed > max_time - 10:
            return None

        size_u = n * m * target_rank
        size_v = m * p * target_rank
        size_w = p * n * target_rank
        total_size = size_u + size_v + size_w

        def residuals(x):
            U = x[:size_u].reshape(n*m, target_rank)
            V = x[size_u:size_u+size_v].reshape(m*p, target_rank)
            W = x[size_u+size_v:].reshape(p*n, target_rank)
            res = np.zeros(num_triples)
            for c in range(target_rank):
                res += U[u_indices, c] * V[v_indices, c] * W[w_indices, c]
            res -= 1.0
            return res

        def jacobian(x):
            U = x[:size_u].reshape(n*m, target_rank)
            V = x[size_u:size_u+size_v].reshape(m*p, target_rank)
            W = x[size_u+size_v:].reshape(p*n, target_rank)
            J = np.zeros((num_triples, total_size))
            for c in range(target_rank):
                J[:, u_indices * target_rank + c] += V[v_indices, c] * W[w_indices, c]
                J[:, size_u + v_indices * target_rank + c] += U[u_indices, c] * W[w_indices, c]
                J[:, size_u + size_v + w_indices * target_rank + c] += U[u_indices, c] * V[v_indices, c]
            return J

        best_norm = float('inf')
        best_x = None

        for attempt in range(max_attempts):
            if time.time() - start_time > max_time - 15:
                break

            if attempt % 4 == 0:
                x0 = np.random.choice([-1, -0.5, 0, 0.5, 1], size=total_size)
            elif attempt % 4 == 1:
                x0 = np.random.uniform(-1.5, 1.5, size=total_size)
            elif attempt % 4 == 2:
                x0 = np.zeros(total_size)
                num_nonzero = max(1, total_size // 5)
                indices = np.random.choice(total_size, num_nonzero, replace=False)
                x0[indices] = np.random.choice([-1, -0.5, 0.5, 1], size=num_nonzero)
            else:
                x0 = np.random.normal(0, 0.5, size=total_size)

            try:
                sol = least_squares(residuals, x0, jac=jacobian, method='lm',
                                   max_nfev=5000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                res_norm = np.linalg.norm(sol.fun)
                if res_norm < best_norm:
                    best_norm = res_norm
                    best_x = sol.x.copy()
                if res_norm < 1e-6:
                    polished = polish_solution(sol.x, target_rank, size_u, size_v, size_w)
                    if polished is not None:
                        return polished
            except Exception:
                continue

        if best_x is not None and best_norm < 1e-4:
            try:
                sol2 = least_squares(residuals, best_x, jac=jacobian, method='lm',
                                    max_nfev=20000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                if np.linalg.norm(sol2.fun) < 1e-8:
                    polished = polish_solution(sol2.x, target_rank, size_u, size_v, size_w)
                    if polished is not None:
                        return polished
            except Exception:
                pass

        return None

    for target_rank in range(16, 25):
        if time.time() - start_time > max_time - 20:
            break
        result = try_rank(target_rank, max_attempts=35)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

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
