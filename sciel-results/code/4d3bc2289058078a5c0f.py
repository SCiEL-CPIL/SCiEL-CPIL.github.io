import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10 seconds margin

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        if R > rank_schoolbook:
            return False
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
                return False
        # Vectorized check
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        T = np.einsum('ijr,jkr,kr->ijk', U_t, V_t, W_t)
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True

    def compute_error(U, V, W):
        R = U.shape[1]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kr->ijk', U_t, V_t, W_t)
        return np.sum((T - 1.0)**2)

    def attempt_rank(R):
        if R >= rank_schoolbook:
            return None
        
        np.random.seed(42 + R)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # ALS with vectorized error check for early stopping
        prev_err = float('inf')
        for it in range(30):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = u
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = v
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = w
            
            err = compute_error(U, V, W)
            if err < 1e-6:
                break
            if err > prev_err * 1.01:
                # Diverging or stuck
                break
            prev_err = err

        if err > 1e-3:
            return None

        # Snap to half-integers
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        
        if verify(U_int, V_int, W_int):
            return U_int, V_int, W_int

        # Local Search on discrete half-integers
        # If the real solution is close, the snapped solution should be close.
        # We try to reduce the error by flipping entries.
        U_cur = U_int.copy()
        V_cur = V_int.copy()
        W_cur = W_int.copy()
        cur_err = compute_error(U_cur, V_cur, W_cur)
        
        improved = True
        max_local_iters = 100
        local_iter = 0
        
        while improved and local_iter < max_local_iters and time.time() < deadline - 5:
            improved = False
            local_iter += 1
            
            # Try flipping entries in U
            for idx in range(U_cur.size):
                # Get linear index to (i, j, r)
                # U is (n*m, R)
                row, col = np.unravel_index(idx, U_cur.shape)
                old_val = U_cur[row, col]
                # Try +0.5 and -0.5
                for delta in [0.5, -0.5]:
                    new_val = old_val + delta
                    U_cur[row, col] = new_val
                    new_err = compute_error(U_cur, V_cur, W_cur)
                    if new_err < cur_err:
                        cur_err = new_err
                        improved = True
                        # Keep the change
                    else:
                        # Revert
                        U_cur[row, col] = old_val
            
            # Try flipping entries in V
            for idx in range(V_cur.size):
                row, col = np.unravel_index(idx, V_cur.shape)
                old_val = V_cur[row, col]
                for delta in [0.5, -0.5]:
                    new_val = old_val + delta
                    V_cur[row, col] = new_val
                    new_err = compute_error(U_cur, V_cur, W_cur)
                    if new_err < cur_err:
                        cur_err = new_err
                        improved = True
                    else:
                        V_cur[row, col] = old_val
                        
            # Try flipping entries in W
            for idx in range(W_cur.size):
                row, col = np.unravel_index(idx, W_cur.shape)
                old_val = W_cur[row, col]
                for delta in [0.5, -0.5]:
                    new_val = old_val + delta
                    W_cur[row, col] = new_val
                    new_err = compute_error(U_cur, V_cur, W_cur)
                    if new_err < cur_err:
                        cur_err = new_err
                        improved = True
                    else:
                        W_cur[row, col] = old_val
            
            if verify(U_cur, V_cur, W_cur):
                return U_cur, V_cur, W_cur

        return None

    # Try ranks 119 down to 115
    for R in range(119, 114, -1):
        res = attempt_rank(R)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
