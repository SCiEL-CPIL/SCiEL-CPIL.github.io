import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36

    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    start_time = time.time()
    time_limit = 110.0

    # Slice-and-sum: partition 9 columns into three 3-column strips
    # For each strip of 3 columns, we need a 4x4 * 4x3 decomposition
    # Use Winograd/Strassen structure for each strip

    # Strategy: Use the known rank-12 decomposition for 4x4 * 4x3
    # This comes from applying Strassen's algorithm with careful blocking

    # Actually, let's construct explicit low-rank decompositions
    # For 4x4 * 4x3, we can use the fact that it's equivalent to
    # computing 3 columns of a 4x4 * 4x4 product

    # Let's try to find a good rank by optimization
    # Start with a reasonable guess and optimize

    best_rank = n * m * p  # schoolbook 144
    best_U, best_V, best_W = None, None, None

    # Try various ranks with optimization
    for R in range(108, 120):  # Try ranks around Strassen-based estimates
        if time.time() - start_time > time_limit:
            break

        for restart in range(3):
            if time.time() - start_time > time_limit:
                break

            np.random.seed(42 + restart * 1000 + R)
            U = np.random.randn(dim_u, R) * 0.1
            V = np.random.randn(dim_v, R) * 0.1
            W = np.random.randn(dim_w, R) * 0.1

            lam = 1e-10
            I_R = np.eye(R) * lam
            converged = False

            for iter in range(80):
                if time.time() - start_time > time_limit:
                    break

                # Update U
                VtV = V.T @ V
                WtW = W.T @ W
                G_U = VtV * WtW + I_R
                R_U = np.einsum('vc,wc,uvw->uc', V, W, T, optimize=True)
                try:
                    U_new = np.linalg.solve(G_U, R_U.T).T
                except np.linalg.LinAlgError:
                    U_new = np.linalg.lstsq(G_U, R_U.T, rcond=None)[0].T
                U = U_new

                # Update V
                UtU = U.T @ U
                G_V = UtU * WtW + I_R
                R_V = np.einsum('uc,wc,uvw->vc', U, W, T, optimize=True)
                try:
                    V_new = np.linalg.solve(G_V, R_V.T).T
                except np.linalg.LinAlgError:
                    V_new = np.linalg.lstsq(G_V, R_V.T, rcond=None)[0].T
                V = V_new

                # Update W
                VtV = V.T @ V
                G_W = UtU * VtV + I_R
                R_W = np.einsum('uc,vc,uvw->wc', U, V, T, optimize=True)
                try:
                    W_new = np.linalg.solve(G_W, R_W.T).T
                except np.linalg.LinAlgError:
                    W_new = np.linalg.lstsq(G_W, R_W.T, rcond=None)[0].T
                W = W_new

                # Check error periodically
                if iter % 15 == 0 or iter == 79:
                    T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-8:
                        # Try rounding to half-integers
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        T_recon_r = np.einsum('uc,vc,wc->uvw', U_rounded, V_rounded, W_rounded, optimize=True)
                        error_r = np.linalg.norm(T_recon_r - T)

                        if error_r < 1e-9:
                            if R < best_rank:
                                best_rank = R
                                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }

                        # Try integer rounding
                        U_rounded_i = np.round(U)
                        V_rounded_i = np.round(V)
                        W_rounded_i = np.round(W)
                        T_recon_ri = np.einsum('uc,vc,wc->uvw', U_rounded_i, V_rounded_i, W_rounded_i, optimize=True)
                        error_ri = np.linalg.norm(T_recon_ri - T)

                        if error_ri < 1e-9:
                            if R < best_rank:
                                best_rank = R
                                best_U, best_V, best_W = U_rounded_i, V_rounded_i, W_rounded_i
                            return {
                                "U": U_rounded_i.tolist(),
                                "V": V_rounded_i.tolist(),
                                "W": W_rounded_i.tolist()
                            }

    # If optimization failed, return schoolbook
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
