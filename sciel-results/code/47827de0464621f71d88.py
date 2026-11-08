import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 135

    n, m, p = DIMENSIONS
    rank22 = 22
    U22 = np.zeros((n * m, rank22))
    V22 = np.zeros((m * p, rank22))
    W22 = np.zeros((p * n, rank22))

    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U22[idx, col] += val
        for idx, val in v_indices:
            V22[idx, col] += val
        for idx, val in w_indices:
            W22[idx, col] += val

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)

    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    target_rank = 21

    def residuals(params):
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank :]

        U_mat = U_flat.reshape(n*m, target_rank)
        V_mat = V_flat.reshape(m*p, target_rank)
        W_mat = W_flat.reshape(p*n, target_rank)

        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(target_rank):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    res.append(val - 1)
        return np.array(res)

    def check_solution(U_mat, V_mat, W_mat, r):
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for rr in range(r):
                        val += U_mat[i*m+j, rr] * V_mat[j*p+k, rr] * W_mat[k*n+i, rr]
                    max_err = max(max_err, abs(val - 1))
        return max_err

    best_err = float('inf')
    best_x = None

    # Try zeroing out each column in turn
    for skip_col in range(rank22):
        if time.time() > deadline:
            break

        cols = [c for c in range(rank22) if c != skip_col]

        U_init = U22[:, cols].flatten()
        V_init = V22[:, cols].flatten()
        W_init = W22[:, cols].flatten()
        x0 = np.concatenate([U_init, V_init, W_init])

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=3000)
            max_err = np.max(np.abs(result.fun))
            if max_err < best_err:
                best_err = max_err
                best_x = result.x
            if max_err < 1e-10:
                break
        except Exception:
            continue

    if best_x is not None and best_err < 1e-8:
        x = best_x
        U_opt = x[:n*m*target_rank].reshape(n*m, target_rank)
        V_opt = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_opt = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)

        # Round to nearest half-integer
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2

        err_round = check_solution(U_round, V_round, W_round, target_rank)
        if err_round < 1e-10:
            U_out = [[float(U_round[i, r]) for r in range(target_rank)] for i in range(n*m)]
            V_out = [[float(V_round[i, r]) for r in range(target_rank)] for i in range(m*p)]
            W_out = [[float(W_round[i, r]) for r in range(target_rank)] for i in range(p*n)]
            return {"U": U_out, "V": V_out, "W": W_out}
        else:
            err_unrounded = check_solution(U_opt, V_opt, W_opt, target_rank)
            if err_unrounded < 1e-10:
                U_out = [[float(U_opt[i, r]) for r in range(target_rank)] for i in range(n*m)]
                V_out = [[float(V_opt[i, r]) for r in range(target_rank)] for i in range(m*p)]
                W_out = [[float(W_opt[i, r]) for r in range(target_rank)] for i in range(p*n)]
                return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback to rank 22
    U_out = [[float(U22[i, r]) for r in range(rank22)] for i in range(n*m)]
    V_out = [[float(V22[i, r]) for r in range(rank22)] for i in range(m*p)]
    W_out = [[float(W22[i, r]) for r in range(rank22)] for i in range(p*n)]
    return {"U": U_out, "V": V_out, "W": W_out}
