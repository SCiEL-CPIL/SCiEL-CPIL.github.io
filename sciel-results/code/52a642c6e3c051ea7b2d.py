import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    # x in 0..7, y in 0..15, z in 0..7
    T = np.zeros((8, 16, 8), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1

    # Current known solution rank 28
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known

        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    U_known_np = np.array(U_known, dtype=float)
    V_known_np = np.array(V_known, dtype=float)
    W_known_np = np.array(W_known, dtype=float)

    def check_solution_exact(U, V, W, rank):
        """Exact verification using Fraction."""
        U_f = np.array(U, dtype=object)
        V_f = np.array(V, dtype=object)
        W_f = np.array(W, dtype=object)
        
        T_est = np.zeros((8, 16, 8), dtype=object)
        for r in range(rank):
            for x in range(8):
                if U_f[x, r] != 0:
                    for y in range(16):
                        if V_f[y, r] != 0:
                            for z in range(8):
                                if W_f[z, r] != 0:
                                    T_est[x, y, z] += U_f[x, r] * V_f[y, r] * W_f[z, r]
        
        for x in range(8):
            for y in range(16):
                for z in range(8):
                    expected = T[x, y, z]
                    actual = T_est[x, y, z]
                    if abs(float(actual) - expected) > 1e-12:
                        return False
        return True

    def check_solution_fast(U_np, V_np, W_np, rank):
        """Fast numerical verification."""
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r]).reshape(8, 16, 8)
        return np.max(np.abs(T_est - T)) < 1e-9

    def compute_residual(U_np, V_np, W_np, rank):
        """Compute squared residual efficiently."""
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r]).reshape(8, 16, 8)
        diff = T_est - T
        return np.sum(diff * diff)

    def compute_residual_fast(U_np, V_np, W_np, rank):
        """Even faster: use einsum."""
        T_est = np.einsum('xr,yr,zr->xyz', U_np[:, :rank], V_np[:, :rank], W_np[:, :rank])
        diff = T_est - T
        return np.sum(diff * diff)

    best_rank = rank_known
    best_U = U_known
    best_V = V_known
    best_W = W_known

    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit

    # Strategy: continuous local search to reduce rank
    # For rank r, start from rank-28 solution with last (28-r) columns zeroed,
    # then do continuous optimization to drive residual to zero.

    def continuous_search(target_rank, U_init, V_init, W_init, max_time=None):
        """Continuous Nelder-Mead style search for exact decomposition at target_rank."""
        if max_time is None:
            max_time = deadline - time.time()
        if max_time < 2:
            return None

        r = target_rank
        # Initialize: take first r columns of known solution, zero the rest
        U = np.zeros((8, r), dtype=float)
        V = np.zeros((16, r), dtype=float)
        W = np.zeros((8, r), dtype=float)
        
        # Copy first r columns from known solution
        U[:, :] = U_known_np[:, :r]
        V[:, :] = V_known_np[:, :r]
        W[:, :] = W_known_np[:, :r]

        # Flatten parameters for optimization
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        n_params = len(params)

        def unpack(p):
            U_p = p[:8*r].reshape(8, r)
            V_p = p[8*r:8*r+16*r].reshape(16, r)
            W_p = p[8*r+16*r:].reshape(8, r)
            return U_p, V_p, W_p

        def objective(p):
            U_p, V_p, W_p = unpack(p)
            return compute_residual_fast(U_p, V_p, W_p, r)

        # Nelder-Mead implementation
        from scipy.optimize import minimize

        # Try multiple starting points with small random perturbations
        best_result = None
        best_obj = objective(params)

        # Initial simplex
        step = 0.1
        simplex = [params.copy()]
        for i in range(n_params):
            p = params.copy()
            p[i] += step
            simplex.append(p)

        # Evaluate initial simplex
        objs = [objective(s) for s in simplex]

        # Simple Nelder-Mead
        for it in range(5000):
            if time.time() > deadline - 5:
                break

            # Sort
            order = np.argsort(objs)
            simplex = [simplex[i] for i in order]
            objs = [objs[i] for i in order]

            if objs[0] < best_obj:
                best_obj = objs[0]
                best_result = simplex[0].copy()

            if best_obj < 1e-15:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]

            # Reflection
            alpha = 1.0
            x_r = centroid + alpha * (centroid - worst)
            f_r = objective(x_r)

            if f_r < objs[0]:
                # Expansion
                gamma = 2.0
                x_e = centroid + gamma * (centroid - worst)
                f_e = objective(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    objs[-1] = f_e
                else:
                    simplex[-1] = x_r
                    objs[-1] = f_r
            elif f_r < objs[-2]:
                simplex[-1] = x_r
                objs[-1] = f_r
            else:
                # Contraction
                rho = 0.5
                x_c = centroid + rho * (worst - centroid)
                f_c = objective(x_c)
                if f_c < objs[-1]:
                    simplex[-1] = x_c
                    objs[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        objs[i] = objective(simplex[i])

        if best_result is None:
            return None

        U_best, V_best, W_best = unpack(best_result)

        # Round to nearest 0.5 values
        U_rounded = np.round(U_best * 2) / 2
        V_rounded = np.round(V_best * 2) / 2
        W_rounded = np.round(W_best * 2) / 2

        # Check if rounded solution is exact
        if check_solution_fast(U_rounded, V_rounded, W_rounded, r):
            # Convert to list format
            U_list = [[float(x) for x in row] for row in U_rounded]
            V_list = [[float(x) for x in row] for row in V_rounded]
            W_list = [[float(x) for x in row] for row in W_rounded]
            return U_list, V_list, W_list

        # Try integer rounding too
        U_int = np.round(U_best)
        V_int = np.round(V_best)
        W_int = np.round(W_best)
        if check_solution_fast(U_int, V_int, W_int, r):
            U_list = [[float(x) for x in row] for row in U_int]
            V_list = [[float(x) for x in row] for row in V_int]
            W_list = [[float(x) for x in row] for row in W_int]
            return U_list, V_list, W_list

        return None

    # Try rank 27 first
    for restart in range(30):
        if time.time() > deadline - 10:
            break

        # Add small random perturbation to initial point for diversity
        if restart == 0:
            U_init = U_known_np[:, :27].copy()
            V_init = V_known_np[:, :27].copy()
            W_init = W_known_np[:, :27].copy()
        else:
            U_init = U_known_np[:, :27].copy() + np.random.randn(8, 27) * 0.05
            V_init = V_known_np[:, :27].copy() + np.random.randn(16, 27) * 0.05
            W_init = W_known_np[:, :27].copy() + np.random.randn(8, 27) * 0.05

        result = continuous_search(27, U_init, V_init, W_init)
        if result is not None:
            U27, V27, W27 = result
            if check_solution_fast(np.array(U27), np.array(V27), np.array(W27), 27):
                best_rank = 27
                best_U, best_V, best_W = U27, V27, W27
                break

    # If rank 27 not found, try rank 26
    if best_rank == 28 and time.time() < deadline - 30:
        for restart in range(20):
            if time.time() > deadline - 10:
                break

            if restart == 0:
                U_init = U_known_np[:, :26].copy()
                V_init = V_known_np[:, :26].copy()
                W_init = W_known_np[:, :26].copy()
            else:
                U_init = U_known_np[:, :26].copy() + np.random.randn(8, 26) * 0.1
                V_init = V_known_np[:, :26].copy() + np.random.randn(16, 26) * 0.1
                W_init = W_known_np[:, :26].copy() + np.random.randn(8, 26) * 0.1

            result = continuous_search(26, U_init, V_init, W_init)
            if result is not None:
                U26, V26, W26 = result
                if check_solution_fast(np.array(U26), np.array(V26), np.array(W26), 26):
                    best_rank = 26
                    best_U, best_V, best_W = U26, V26, W26
                    break

    return {"U": best_U, "V": best_V, "W": best_W}
