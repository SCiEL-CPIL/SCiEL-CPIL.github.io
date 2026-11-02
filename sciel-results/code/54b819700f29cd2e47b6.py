import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Construct the baseline Rank 24 solution
    rank_base = n * m * p
    U_base = np.zeros((n * m, rank_base), dtype=float)
    V_base = np.zeros((m * p, rank_base), dtype=float)
    W_base = np.zeros((p * n, rank_base), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j, col] = 1.0
                V_base[j * p + k, col] = 1.0
                W_base[k * n + i, col] = 1.0
                col += 1
                
    best_result = {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    best_rank = rank_base

    # 2. Define the target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    def try_rank_als(r, max_restarts=30, max_iters=50):
        """
        Attempt to find a valid decomposition for a specific rank r using ALS.
        """
        if time.time() > deadline:
            return None
            
        U_shape = (n * m, r)
        V_shape = (m * p, r)
        W_shape = (p * n, r)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            np.random.seed(1234 + restart * 999)
            U = np.random.randn(*U_shape) * 0.5
            V = np.random.randn(*V_shape) * 0.5
            W = np.random.randn(*W_shape) * 0.5
            
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                    
                # Update U
                A_U = np.einsum('jr,kr->jkr', V, W).reshape(-1, r)
                U_new = np.zeros_like(U)
                for i in range(n * m):
                    b = target[i, :, :].reshape(-1)
                    try:
                        U_new[i, :] = np.linalg.lstsq(A_U, b, rcond=None)[0]
                    except:
                        U_new[i, :] = U[i, :]
                U = U_new
                
                # Update V
                A_V = np.einsum('ir,kr->ikr', U, W).reshape(-1, r)
                V_new = np.zeros_like(V)
                for j in range(m * p):
                    b = target[:, j, :].reshape(-1)
                    try:
                        V_new[j, :] = np.linalg.lstsq(A_V, b, rcond=None)[0]
                    except:
                        V_new[j, :] = V[j, :]
                V = V_new
                
                # Update W
                A_W = np.einsum('ir,jr->ijr', U, V).reshape(-1, r)
                W_new = np.zeros_like(W)
                for k in range(p * n):
                    b = target[:, :, k].reshape(-1)
                    try:
                        W_new[k, :] = np.linalg.lstsq(A_W, b, rcond=None)[0]
                    except:
                        W_new[k, :] = W[k, :]
                W = W_new
                
                # Calculate error
                T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.sum((T_approx - target) ** 2)
                
                if abs(err - prev_err) < 1e-9:
                    break
                prev_err = err
                
                # Check for convergence to exact solution
                if err < 1e-5:
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    T_approx_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    if np.allclose(T_approx_r, target, atol=1e-9):
                        return U_r, V_r, W_r

        return None

    # Try ranks from low to high
    for r in range(11, 24):
        if time.time() > deadline - 5:
            break
            
        sol = try_rank_als(r)
        if sol:
            U, V, W = sol
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = r
            break
            
    return best_result
