import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def verify_full(U, V, W, T, tol=1e-10):
        T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(T_check - T))

    # --- Rank-55 decomposition: j=0 separately + Strassen for j=1,2 ---
    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0

        # Part A: j=0 (first column of A, first row of B)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            W[w_11, col] += 1
            col += 1

            # M2 = (a21 + a22)b11
            U[u_21, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1
            W[w_11, col] += -1
            col += 1

            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1
            V[v_22, col] += -1
            W[w_01, col] += 1
            W[w_11, col] += 1
            col += 1

            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1
            V[v_11, col] += -1
            W[w_00, col] += 1
            W[w_10, col] += 1
            col += 1

            # M5 = (a11 + a12)b22
            U[u_11, col] += 1
            U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1
            W[w_01, col] += 1
            col += 1

            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1
            U[u_11, col] += -1
            V[v_11, col] += 1
            V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1

            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1
            U[u_22, col] += -1
            V[v_21, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1

        return U, V, W, rank

    U_best, V_best, W_best, rank_best = rank55_decomposition()
    err = verify(U_best, V_best, W_best)

    # --- Improved ALS exploiting block structure ---
    def block_als(U, V, W, r, max_iters=60, tol=1e-12):
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for k in range(p):
                    M_sub = np.zeros((m, r))
                    for j in range(m):
                        M_sub[j, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        sol = np.linalg.lstsq(M_sub, np.ones(m), rcond=None)[0]
                        for j in range(m):
                            U_new[i * m + j, :] = sol
                    except:
                        break
            U = U_new

            # Solve for V
            V_new = np.zeros_like(V)
            for i in range(n):
                for k in range(p):
                    N_sub = np.zeros((m, r))
                    for j in range(m):
                        N_sub[j, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        sol = np.linalg.lstsq(N_sub, np.ones(m), rcond=None)[0]
                        for j in range(m):
                            V_new[j * p + k, :] = sol
                    except:
                        break
            V = V_new

            # Solve for W
            W_new = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    P_sub = np.zeros((m, r))
                    for j in range(m):
                        P_sub[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        sol = np.linalg.lstsq(P_sub, np.ones(m), rcond=None)[0]
                        for j in range(m):
                            W_new[k * n + i, :] = sol
                    except:
                        break
            W = W_new

            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        e = abs(val - 1.0)
                        if e > max_err:
                            max_err = e
            if max_err < tol:
                break

        return U, V, W

    # --- Full ALS (from inspiration 1) ---
    def full_als(U, V, W, r, T, max_iters=30, tol=1e-12):
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Solve for U
            M = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, r)
            try:
                U = T.reshape(nm, mp * pn) @ np.linalg.pinv(M.T)
            except:
                break

            # Solve for V
            N = np.einsum('ir,kr->ikr', U, W).reshape(nm * pn, r)
            try:
                V = T.reshape(mp, nm * pn) @ np.linalg.pinv(N.T)
            except:
                break

            # Solve for W
            P = np.einsum('ir,jr->ijr', U, V).reshape(nm * mp, r)
            try:
                W = T.reshape(pn, nm * mp) @ np.linalg.pinv(P.T)
            except:
                break

            T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.max(np.abs(T_check - T)) < tol:
                break

        return U, V, W

    # --- Grid search for half-integer solutions ---
    def grid_search_rank(r, T, max_trials=5000):
        """
        For small rank reductions, try to find half-integer solutions
        by exploring structured perturbations of the rank-55 solution.
        """
        if r >= 55:
            return None

        # The idea: we need to remove (55 - r) columns from the rank-55 solution
        # and adjust the remaining columns to compensate.
        # For r=54, we remove 1 column and adjust.
        # For r=53, we remove 2 columns and adjust.
        
        # Strategy: try removing each column and see if the remaining system
        # has a solution with half-integer entries.
        
        # This is complex, so we use a different approach:
        # Random search with strong bias toward half-integers.
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Initialize near rank-55
            U_c = U_best[:, :r].copy() + 0.1 * np.random.randn(nm, r)
            V_c = V_best[:, :r].copy() + 0.1 * np.random.randn(mp, r)
            W_c = W_best[:, :r].copy() + 0.1 * np.random.randn(pn, r)
            
            # Run a few ALS iterations
            U_c, V_c, W_c = block_als(U_c, V_c, W_c, r, max_iters=15)
            
            # Try to round to half-integers
            U_r = np.round(U_c * 2) / 2
            V_r = np.round(V_c * 2) / 2
            W_r = np.round(W_c * 2) / 2
            
            err = verify_full(U_r, V_r, W_r, T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            if err < 1e-10:
                return best_U, best_V, best_W
        
        if best_err < 1e-10:
            return best_U, best_V, best_W
        return None

    # --- Try to reduce rank ---
    if time.time() < deadline - 10:
        # Try ranks from 54 down
        for r in range(rank_best - 1, max(nm, mp, pn) - 1, -1):
            if time.time() > deadline - 5:
                break

            found = False
            max_restarts = 30 if r >= 50 else 20

            for trial in range(max_restarts):
                if time.time() > deadline - 5:
                    break

                np.random.seed(trial * 1000 + r)

                # Initialize near the rank-55 solution
                if r <= rank_best:
                    U_c = U_best[:, :r].copy() + 0.05 * np.random.randn(nm, r)
                    V_c = V_best[:, :r].copy() + 0.05 * np.random.randn(mp, r)
                    W_c = W_best[:, :r].copy() + 0.05 * np.random.randn(pn, r)
                else:
                    U_c = np.random.randn(nm, r)
                    V_c = np.random.randn(mp, r)
                    W_c = np.random.randn(pn, r)

                # Run block ALS
                U_c, V_c, W_c = block_als(U_c, V_c, W_c, r, max_iters=50)

                # Check convergence
                T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
                error = np.max(np.abs(T_check - T))

                if error < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U_c * 2) / 2
                    V_r = np.round(V_c * 2) / 2
                    W_r = np.round(W_c * 2) / 2

                    T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    if np.max(np.abs(T_check2 - T)) < 1e-10:
                        U_best, V_best, W_best, rank_best = U_r, V_r, W_r, r
                        found = True
                        break
                    else:
                        # Use unrounded if close enough
                        if error < 1e-10:
                            U_best, V_best, W_best, rank_best = U_c, V_c, W_c, r
                            found = True
                            break

            if found:
                break

        # If we haven't found a lower rank, try grid search for r=54
        if rank_best == 55 and time.time() < deadline - 10:
            result = grid_search_rank(54, T, max_trials=2000)
            if result is not None:
                U_best, V_best, W_best, rank_best = result[0], result[1], result[2], 54

    # Final verification
    final_err = verify(U_best, V_best, W_best)
    if final_err > 1e-8:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        U_best, V_best, W_best = U, V, W

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
