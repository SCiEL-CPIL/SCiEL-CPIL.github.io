import numpy as np
import itertools
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def is_valid_decomposition(U, V, W, T):
        try:
            decomposed = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(decomposed - T)
            return error, decomposed
        except Exception:
            return float('inf'), None

    def snap_to_half(x):
        return np.round(x * 2.0) / 2.0

    # Try different ranks
    # HOOI is more robust, so we can try fewer random restarts
    for R in range(12, 25):
        for _ in range(20):
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # HOOI Iterations
            # We want to minimize || T - U x V x W ||_F
            # Step 1: Update U
            # M_U[a, r] = sum_{b,c} T[a,b,c] V[b, r'] W[c, r''] ?
            # No, HOOI projects T onto the subspace spanned by V and W.
            # The update for U is:
            # U_new = SVD( T x_2 V x_3 W )_U
            # Where T x_2 V is a tensor of size (nm, R, pn)
            # And (T x_2 V) x_3 W is a matrix of size (nm, R)
            
            for it in range(100):
                # Update U
                # T x_2 V: (nm, R, pn)
                # Temp_U = np.einsum('abc, br -> acr', T, V)
                # M_U = np.einsum('acr, cr -> ar', Temp_U, W)
                Temp_U = np.einsum('abc, br -> acr', T, V)
                M_U = np.einsum('acr, cr -> ar', Temp_U, W)
                
                # SVD of M_U
                # M_U is (nm, R). We want U (nm, R) such that M_U approx U @ diag(s)
                # So U is the left singular vectors.
                U_svd, _, _ = np.linalg.svd(M_U, full_matrices=False)
                # Ensure deterministic sign
                for i in range(R):
                    if U_svd[:, i].max() < 0:
                        U_svd[:, i] *= -1
                U = U_svd[:, :R]
                
                # Update V
                # T x_1 U: (R, mp, pn)
                # Temp_V = np.einsum('abc, ar -> bcr', T, U)
                # M_V = np.einsum('bcr, cr -> br', Temp_V, W)
                Temp_V = np.einsum('abc, ar -> bcr', T, U)
                M_V = np.einsum('bcr, cr -> br', Temp_V, W)
                
                V_svd, _, _ = np.linalg.svd(M_V, full_matrices=False)
                for i in range(R):
                    if V_svd[:, i].max() < 0:
                        V_svd[:, i] *= -1
                V = V_svd[:, :R]
                
                # Update W
                # T x_1 U: (R, mp, pn)
                # Temp_W = np.einsum('abc, ar -> bcr', T, U)
                # M_W = np.einsum('bcr, br -> cr', Temp_W, V)
                Temp_W = np.einsum('abc, ar -> bcr', T, U)
                M_W = np.einsum('bcr, br -> cr', Temp_W, V)
                
                W_svd, _, _ = np.linalg.svd(M_W, full_matrices=False)
                for i in range(R):
                    if W_svd[:, i].max() < 0:
                        W_svd[:, i] *= -1
                W = W_svd[:, :R]
                
                # Check convergence
                error, _ = is_valid_decomposition(U, V, W, T)
                if error < 1e-10:
                    break

            error, _ = is_valid_decomposition(U, V, W, T)
            if error < 1e-8:
                # Snap to half-integers
                U_snap = snap_to_half(U)
                V_snap = snap_to_half(V)
                W_snap = snap_to_half(W)
                
                error_snap, _ = is_valid_decomposition(U_snap, V_snap, W_snap, T)
                if error_snap < 1e-8:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                
                # Try integer snap
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                error_int, _ = is_valid_decomposition(U_int, V_int, W_int, T)
                if error_int < 1e-8:
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }

    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
