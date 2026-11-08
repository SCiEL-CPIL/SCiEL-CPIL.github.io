import numpy as np
import time
from scipy.optimize import least_squares, minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8

    # Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) valid
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # Precompute index triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)  # 24

    def build_residuals(U, V, W):
        """Vectorized residual computation."""
        # U: (nm, r), V: (mp, r), W: (pn, r)
        r = U.shape[1]
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        res = np.zeros(num_triples)
        for idx, (i, j, k) in enumerate(triples):
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            res[idx] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :]) - 1.0
        return res

    def compute_max_err(U, V, W):
        r = U.shape[1]
        max_err = 0.0
        for idx, (i, j, k) in enumerate(triples):
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        return max_err

    # Rank 22 Baseline Construction
    rank22 = 22
    U22 = np.zeros((nm, rank22))
    V22 = np.zeros((mp, rank22))
    W22 = np.zeros((pn, rank22))

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
    best_err = float('inf')
    best_x = None
    found_rank21 = False

    # Strategy 1: Remove each column from Rank 22 and optimize with least_squares
    for skip_col in range(rank22):
        if time.time() > deadline - 30:
            break

        cols = [c for c in range(rank22) if c != skip_col]

        U_init = U22[:, cols].copy()
        V_init = V22[:, cols].copy()
        W_init = W22[:, cols].copy()
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

        def residuals_ls(params):
            U_flat = params[:nm * target_rank]
            V_flat = params[nm * target_rank: nm * target_rank + mp * target_rank]
            W_flat = params[nm * target_rank + mp * target_rank:]
            U_mat = U_flat.reshape(nm, target_rank)
            V_mat = V_flat.reshape(mp, target_rank)
            W_mat = W_flat.reshape(pn, target_rank)
            return build_residuals(U_mat, V_mat, W_mat)

        try:
            result = least_squares(residuals_ls, x0, method='lm', max_nfev=10000,
                                   ftol=1e-15, xtol=1e-15, gtol=1e-15)
            max_err = np.max(np.abs(result.fun))

            if max_err < best_err:
                best_err = max_err
                best_x = result.x.copy()

            if max_err < 1e-10:
                found_rank21 = True
                break

        except Exception:
            continue

    # Strategy 2: Smooth surrogate - minimize sum of squares with tighter tol
    if not found_rank21 and time.time() < deadline - 20:
        for skip_col in range(rank22):
            if time.time() > deadline - 10:
                break

            cols = [c for c in range(rank22) if c != skip_col]
            U_init = U22[:, cols].copy()
            V_init = V22[:, cols].copy()
            W_init = W22[:, cols].copy()
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            def residuals_ls2(params):
                U_flat = params[:nm * target_rank]
                V_flat = params[nm * target_rank: nm * target_rank + mp * target_rank]
                W_flat = params[nm * target_rank + mp * target_rank:]
                U_mat = U_flat.reshape(nm, target_rank)
                V_mat = V_flat.reshape(mp, target_rank)
                W_mat = W_flat.reshape(pn, target_rank)
                return build_residuals(U_mat, V_mat, W_mat)

            try:
                # Use trf method which handles larger problems better
                result = least_squares(residuals_ls2, x0, method='trf', max_nfev=20000,
                                       ftol=1e-14, xtol=1e-14, gtol=1e-14)
                max_err = np.max(np.abs(result.fun))

                if max_err < best_err:
                    best_err = max_err
                    best_x = result.x.copy()

                if max_err < 1e-10:
                    found_rank21 = True
                    break

            except Exception:
                continue

    # Strategy 3: Random perturbations around rank-22 solutions
    if not found_rank21 and time.time() < deadline - 15:
        rng = np.random.RandomState(42)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break

            # Pick a random skip column and add small random perturbation
            skip_col = rng.randint(rank22)
            cols = [c for c in range(rank22) if c != skip_col]
            U_init = U22[:, cols].copy()
            V_init = V22[:, cols].copy()
            W_init = W22[:, cols].copy()

            # Add small random perturbation to escape local minima
            scale = 0.01
            U_init += rng.normal(0, scale, U_init.shape)
            V_init += rng.normal(0, scale, V_init.shape)
            W_init += rng.normal(0, scale, W_init.shape)

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            def residuals_ls3(params):
                U_flat = params[:nm * target_rank]
                V_flat = params[nm * target_rank: nm * target_rank + mp * target_rank]
                W_flat = params[nm * target_rank + mp * target_rank:]
                U_mat = U_flat.reshape(nm, target_rank)
                V_mat = V_flat.reshape(mp, target_rank)
                W_mat = W_flat.reshape(pn, target_rank)
                return build_residuals(U_mat, V_mat, W_mat)

            try:
                result = least_squares(residuals_ls3, x0, method='lm', max_nfev=5000,
                                       ftol=1e-14, xtol=1e-14, gtol=1e-14)
                max_err = np.max(np.abs(result.fun))

                if max_err < best_err:
                    best_err = max_err
                    best_x = result.x.copy()

                if max_err < 1e-10:
                    found_rank21 = True
                    break

            except Exception:
                continue

    # Strategy 4: High-precision polish if we got close
    if best_x is not None and best_err < 0.1 and not found_rank21 and time.time() < deadline - 5:
        x = best_x.copy()

        def residuals_polish(params):
            U_flat = params[:nm * target_rank]
            V_flat = params[nm * target_rank: nm * target_rank + mp * target_rank]
            W_flat = params[nm * target_rank + mp * target_rank:]
            U_mat = U_flat.reshape(nm, target_rank)
            V_mat = V_flat.reshape(mp, target_rank)
            W_mat = W_flat.reshape(pn, target_rank)
            return build_residuals(U_mat, V_mat, W_mat)

        try:
            result = least_squares(residuals_polish, x, method='lm', max_nfev=50000,
                                   ftol=1e-16, xtol=1e-16, gtol=1e-16)
            max_err = np.max(np.abs(result.fun))
            if max_err < best_err:
                best_err = max_err
                best_x = result.x.copy()
            if max_err < 1e-10:
                found_rank21 = True
        except Exception:
            pass

    # Check if we found a valid Rank 21 solution
    if best_x is not None and best_err < 1e-8:
        x = best_x
        U_opt = x[:nm * target_rank].reshape(nm, target_rank)
        V_opt = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
        W_opt = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

        # Try rounding to nearest half-integer
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2

        err_round = compute_max_err(U_round, V_round, W_round)
        if err_round < 1e-10:
            U_out = [[float(U_round[i, r]) for r in range(target_rank)] for i in range(nm)]
            V_out = [[float(V_round[i, r]) for r in range(target_rank)] for i in range(mp)]
            W_out = [[float(W_round[i, r]) for r in range(target_rank)] for i in range(pn)]
            return {"U": U_out, "V": V_out, "W": W_out}
        else:
            # Check unrounded
            err_unrounded = compute_max_err(U_opt, V_opt, W_opt)
            if err_unrounded < 1e-10:
                U_out = [[float(U_opt[i, r]) for r in range(target_rank)] for i in range(nm)]
                V_out = [[float(V_opt[i, r]) for r in range(target_rank)] for i in range(mp)]
                W_out = [[float(W_opt[i, r]) for r in range(target_rank)] for i in range(pn)]
                return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback to Rank 22
    U_out = [[float(U22[i, r]) for r in range(rank22)] for i in range(nm)]
    V_out = [[float(V22[i, r]) for r in range(rank22)] for i in range(mp)]
    W_out = [[float(W22[i, r]) for r in range(rank22)] for i in range(pn)]
    return {"U": U_out, "V": V_out, "W": W_out}
