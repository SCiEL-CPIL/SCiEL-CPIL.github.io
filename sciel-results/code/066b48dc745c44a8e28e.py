import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_ranks = [23, 24, 25, 26, 27]
    
    # Build the tensor T
    # T is of shape (n*m, m*p, p*n)
    # Flatten indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    time_limit = 100  # Leave 20s for safety
    
    # Function to check if a decomposition is valid and returns integer/half-integer
    def check_decomposition(U, V, W, T):
        # Reconstruct T from U, V, W
        # T_recon[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Efficiently: T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # Check exactness
        if not np.allclose(T_recon, T, atol=1e-9):
            return False
        # Check entries are integer or half-integer
        # Multiply by 2 and check if integer
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if not np.allclose(U2, np.round(U2), atol=1e-9):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-9):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-9):
            return False
        return True

    # Try to find a decomposition of rank R
    def find_decomposition(R, T, max_iters=1000):
        # Random initialization
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Simple gradient descent or ALS (Alternating Least Squares)
        # ALS is often faster for tensor decomposition
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                return None
            
            # Update U: minimize || T - sum U[:, r] outer V[:, r] outer W[:, r] ||
            # T_unfolded1 (size nm x mp*pn) = U @ (V * W) ... 
            # Let's use the property: T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
            # For fixed V, W, U_{ir} = (T_{ijk} * V_{jr} * W_{kr}) / (V_{jr}^2 * W_{kr}^2) ... 
            # Actually, linear least squares:
            # T_{i, j, k} = sum_r U_{i, r} (V_{j, r} W_{k, r})
            # Let M_r = V[:, r] * W[:, r] (element-wise product, size mp*pn)
            # Then T_unfolded1[i, (j, k)] = sum_r U[i, r] M_r[j, k]
            # So U = T_unfolded1 @ (M^T @ M)^-1
            
            # Construct M: shape (R, mp*pn)
            M = np.zeros((R, m*p * p*n))
            for r in range(R):
                M[r, :] = (V[:, r][:, None] * W[:, r][None, :]).flatten()
            
            T1 = T.reshape(n*m, m*p * p*n)
            
            # Solve U = T1 @ (M^T M)^-1
            # Use pseudo-inverse for stability
            try:
                U = T1 @ np.linalg.pinv(M.T @ M) @ M.T
            except np.linalg.LinAlgError:
                pass

            # Update V
            M_v = np.zeros((R, n*m * p*n))
            for r in range(R):
                M_v[r, :] = (U[:, r][:, None] * W[:, r][None, :]).flatten()
            T2 = T.reshape(m*p, n*m * p*n)
            try:
                V = T2 @ np.linalg.pinv(M_v.T @ M_v) @ M_v.T
            except np.linalg.LinAlgError:
                pass

            # Update W
            M_w = np.zeros((R, n*m * m*p))
            for r in range(R):
                M_w[r, :] = (U[:, r][:, None] * V[:, r][None, :]).flatten()
            T3 = T.reshape(p*n, n*m * m*p)
            try:
                W = T3 @ np.linalg.pinv(M_w.T @ M_w) @ M_w.T
            except np.linalg.LinAlgError:
                pass
            
            # Check convergence
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_recon, T, atol=1e-6):
                # Try to round to integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if check_decomposition(U_rounded, V_rounded, W_rounded, T):
                    return U_rounded, V_rounded, W_rounded

        return None

    # Try ranks in increasing order
    for R in target_ranks:
        if time.time() - start_time > time_limit:
            break
        
        # Multiple restarts for each rank
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
            
            result = find_decomposition(R, T, max_iters=500)
            if result is not None:
                U, V, W = result
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}

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
