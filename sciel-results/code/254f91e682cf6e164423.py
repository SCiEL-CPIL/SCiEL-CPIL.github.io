import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    schoolbook_rank = n * m * p
    min_rank = max(n * m, m * p, p * n)
    
    def compute_als_matrices(V, W):
        """
        Precompute A_ij and b_ij for U update.
        A_ij is R x R, b_ij is R.
        A_ij[r, r'] = sum_k V[j,k,r] * W[k,i,r] * V[j,k,r'] * W[k,i,r']
        b_ij[r] = sum_k V[j,k,r] * W[k,i,r]
        """
        R = V.shape[2]
        A = np.zeros((n, m, R, R))
        b = np.zeros((n, m, R))
        
        for i in range(n):
            W_i = W[:, i, :]
            for j in range(m):
                V_j = V[j, :, :]
                Z = V_j * W_i
                b[i, j, :] = np.sum(Z, axis=0)
                A[i, j, :, :] = np.sum(Z[:, :, np.newaxis] * Z[:, np.newaxis, :], axis=0)
        
        return A, b

    def als_step(U, V, W, R, max_iters=100, tol=1e-8):
        """
        Perform ALS optimization.
        U: (n, m, R)
        V: (m, p, R)
        W: (p, n, R)
        """
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            A_u, b_u = compute_als_matrices(V, W)
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    try:
                        U_new[i, j, :] = np.linalg.solve(A_u[i, j, :, :] + 1e-8 * np.eye(R), b_u[i, j, :])
                    except np.linalg.LinAlgError:
                        U_new[i, j, :] = np.linalg.lstsq(A_u[i, j, :, :], b_u[i, j, :])[0]
            U = U_new
            
            # Update V
            A_v = np.zeros((m, p, R, R))
            b_v = np.zeros((m, p, R))
            for j in range(m):
                U_j = U[:, j, :]
                for k in range(p):
                    W_k = W[k, :, :]
                    Z = U_j * W_k
                    b_v[j, k, :] = np.sum(Z, axis=0)
                    A_v[j, k, :, :] = np.sum(Z[:, :, np.newaxis] * Z[:, np.newaxis, :], axis=0)
            
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    try:
                        V_new[j, k, :] = np.linalg.solve(A_v[j, k, :, :] + 1e-8 * np.eye(R), b_v[j, k, :])
                    except np.linalg.LinAlgError:
                        V_new[j, k, :] = np.linalg.lstsq(A_v[j, k, :, :], b_v[j, k, :])[0]
            V = V_new
            
            # Update W
            A_w = np.zeros((p, n, R, R))
            b_w = np.zeros((p, n, R))
            for k in range(p):
                V_k = V[:, k, :]
                for i in range(n):
                    U_i = U[i, :, :]
                    Z = U_i * V_k
                    b_w[k, i, :] = np.sum(Z, axis=0)
                    A_w[k, i, :, :] = np.sum(Z[:, :, np.newaxis] * Z[:, np.newaxis, :], axis=0)
            
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    try:
                        W_new[k, i, :] = np.linalg.solve(A_w[k, i, :, :] + 1e-8 * np.eye(R), b_w[k, i, :])
                    except np.linalg.LinAlgError:
                        W_new[k, i, :] = np.linalg.lstsq(A_w[k, i, :, :], b_w[k, i, :])[0]
            W = W_new
            
            # Calculate error
            tensor_val = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            err = np.sum((tensor_val - 1.0) ** 2)
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_error < tol:
                break
                
        return best_U, best_V, best_W, best_error

    def try_rank(R, max_restarts=5):
        """Try to find a decomposition of rank R"""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            np.random.seed(42 + R * 10 + restart)
            U = np.random.randn(n, m, R) * 0.5
            V = np.random.randn(m, p, R) * 0.5
            W = np.random.randn(p, n, R) * 0.5
            
            # Run ALS
            U_opt, V_opt, W_opt, err = als_step(U, V, W, R, max_iters=50, tol=1e-6)
            
            if err < 1e-4:
                # Check if it's exact enough to round
                # Round to half-integers
                U_int = np.round(U_opt * 2) / 2
                V_int = np.round(V_opt * 2) / 2
                W_int = np.round(W_opt * 2) / 2
                
                # Verify exactness
                tensor_val = np.einsum('ijr,jkr,kir->ijk', U_int, V_int, W_int)
                if np.allclose(tensor_val, 1.0, atol=1e-9):
                    # Convert to required format
                    U_list = U_int.reshape(n * m, R).tolist()
                    V_list = V_int.reshape(m * p, R).tolist()
                    W_list = W_int.reshape(p * n, R).tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from lower bound upward
    for rank_offset in range(0, 60):
        if time.time() > deadline:
            break
        
        R = min_rank + rank_offset
        result = try_rank(R, max_restarts=3)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
