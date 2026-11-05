"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    Uses vectorized ALS with integer projection.
    """
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=float)
    
    def compute_error(U, V, W):
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i, j, :], V[j, k, :] * W[k, i, :])
                    err += (val - 1.0) ** 2
        return np.sqrt(err)

    def project_to_half_int(U, V, W):
        return np.round(2 * U) / 2.0, np.round(2 * V) / 2.0, np.round(2 * W) / 2.0

    def als_decompose(r, n_iter=200, n_restarts=5):
        best_err = float('inf')
        best_sol = None
        
        for restart in range(n_restarts):
            U = np.random.randn(n, m, r) * 0.1
            V = np.random.randn(m, p, r) * 0.1
            W = np.random.randn(p, n, r) * 0.1
            
            for it in range(n_iter):
                # Update U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        A = V[j, :, :] * W[:, i, :]  # (p, r)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A.T, T[i, j, :], rcond=None)
                            U_new[i, j, :] = sol
                        except:
                            U_new[i, j, :] = U[i, j, :]
                U = U_new
                
                # Update V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        B = U[:, j, :] * W[k, :, :]  # (n, r)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(B.T, T[:, j, k], rcond=None)
                            V_new[j, k, :] = sol
                        except:
                            V_new[j, k, :] = V[j, k, :]
                V = V_new
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        C = U[i, :, :] * V[:, k, :]  # (m, r)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(C.T, T[i, :, k], rcond=None)
                            W_new[k, i, :] = sol
                        except:
                            W_new[k, i, :] = W[k, i, :]
                W = W_new
                
                # Check convergence
                if it % 10 == 0:
                    err = compute_error(U, V, W)
                    if err < 1e-8:
                        break
            
            # Project and verify
            U_p, V_p, W_p = project_to_half_int(U, V, W)
            err_p = compute_error(U_p, V_p, W_p)
            if err_p < 1e-6:
                return U_p, V_p, W_p
            if err_p < best_err:
                best_err = err_p
                best_sol = (U_p, V_p, W_p)
                
        return None

    # Try ranks from 60 to 71
    for r in range(60, 72):
        res = als_decompose(r, n_iter=100, n_restarts=10)
        if res is not None:
            U, V, W = res
            U_list = U.reshape(n * m, r).tolist()
            V_list = V.reshape(m * p, r).tolist()
            W_list = W.reshape(p * n, r).tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
