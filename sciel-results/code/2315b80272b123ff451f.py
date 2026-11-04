import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = 3, 3, 4
    N, M, P = n * m, m * p, p * n  # 9, 12, 12

    # Build target tensor T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_flat = T.reshape(N * M, P)  # (108, 12)
    T_flat_U = T.reshape(M * P, N)  # (144, 9)
    T_flat_V = T.reshape(N * P, M)  # (108, 12)

    best_rank = 36
    best_U = None
    best_V = None
    best_W = None

    # Use vectorized ALS for speed
    def als_step(U, V, W, R):
        """One ALS iteration, vectorized."""
        # B[r] = vec(U[:,r] ⊗ V[:,r]) for all r
        # B has shape (N*M, R)
        B = np.einsum('ar,br->abr', U, V).reshape(N * M, R)
        W_new = np.linalg.lstsq(B, T_flat, rcond=None)[0].T

        C = np.einsum('br,cr->bcr', V, W_new).reshape(M * P, R)
        U_new = np.linalg.lstsq(C, T_flat_U, rcond=None)[0].T

        D = np.einsum('ar,cr->acr', U_new, W_new).reshape(N * P, R)
        V_new = np.linalg.lstsq(D, T_flat_V, rcond=None)[0].T

        return U_new, V_new, W_new

    # Try ranks from a lower bound upward
    for R in range(24, 37):
        if time.time() > deadline:
            break

        found = False
        n_attempts = 80 if R <= 28 else 40

        for attempt in range(n_attempts):
            if time.time() > deadline:
                break

            np.random.seed(7919 * R + attempt)

            # Structured initialization
            U = np.random.randn(N, R) * 0.2
            V = np.random.randn(M, R) * 0.2
            W = np.random.randn(P, R) * 0.2

            # Add index-based structure
            for r in range(R):
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, r] += 0.4 * ((i + 2*j + r) % 3 - 1)
                for j in range(m):
                    for k in range(p):
                        V[j * p + k, r] += 0.4 * ((j + 2*k + r) % 3 - 1)
                for k in range(p):
                    for i in range(n):
                        W[k * n + i, r] += 0.4 * ((k + 2*i + r) % 4 - 1.5)

            # ALS iterations
            converged = False
            for it in range(40):
                if time.time() > deadline:
                    break

                U, V, W = als_step(U, V, W, R)

                T_pred = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_pred - T)

                if error < 1e-9:
                    converged = True
                    break

            if converged:
                best_rank = R
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break

        if found:
            break

    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook decomposition (rank = n*m*p = 36)
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
