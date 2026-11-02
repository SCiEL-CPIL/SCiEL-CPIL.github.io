import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36

    # Construct the target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    # Strategy: Use Winograd-style block decomposition
    # Partition 4x4 into four 2x2 blocks, 4x9 into four 2x3 blocks
    # Apply rank-7 Winograd to each 2x2 * 2x3 block
    # Total rank: 16 blocks * 7 = 112

    # First, let's try to construct a Winograd-based decomposition
    # For 2x2 * 2x3 matrix multiplication, Winograd uses 7 multiplications

    # Winograd 2x2 * 2x2 has rank 7 (Strassen)
    # For 2x2 * 2x3, we can extend this

    # Actually, let's think about this differently.
    # The 4x4 * 4x9 multiplication can be seen as:
    # [A11 A12] [B11 B12 B13] = [C11 C12 C13]
    # [A21 A22] [B21 B22 B23]   [C21 C22 C23]
    # where Aij are 2x2, Bij are 2x3, Cij are 2x3

    # Each Cij = Ai1*B1j + Ai2*B2j (sum of two 2x2 * 2x3 products)

    # For 2x2 * 2x3, we need at least 6 multiplications (2*3)
    # But with Winograd/Strassen tricks, we might do better

    # Let's try a different approach: use the known rank bounds
    # For 4x4 * 4x4, Strassen gives rank 49
    # For 4x4 * 4x9, we can extend: each 4x4 * 4x1 column needs rank 7
    # So 9 columns * 7 = 63? No, that's not right either.

    # Actually, let's use the tensor rank approach with better initialization
    # and more careful optimization

    # Try ranks from 112 down to 90
    for R in range(112, 89, -1):
        if time.time() - start_time > time_limit:
            break

        # Try random restarts
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break

            np.random.seed(12345 + restart * 7919 + R)
            U = np.random.randn(dim_u, R) * 0.1
            V = np.random.randn(dim_v, R) * 0.1
            W = np.random.randn(dim_w, R) * 0.1

            # Use alternating least squares with better convergence
            lam = 1e-10
            I_R = np.eye(R) * lam

            converged = False
            for iter in range(100):
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

                # Check error
                if iter % 10 == 0 or iter == 99:
                    T_recon = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-8:
                        # Try to round to half-integers
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2

                        T_recon_r = np.einsum('uc,vc,wc->uvw', U_rounded, V_rounded, W_rounded, optimize=True)
                        error_r = np.linalg.norm(T_recon_r - T)

                        if error_r < 1e-9:
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
                            return {
                                "U": U_rounded_i.tolist(),
                                "V": V_rounded_i.tolist(),
                                "W": W_rounded_i.tolist()
                            }

    # If no better rank found, return schoolbook
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
