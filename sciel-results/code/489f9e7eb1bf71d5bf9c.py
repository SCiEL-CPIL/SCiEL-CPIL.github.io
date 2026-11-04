"""Program to find rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = DIMENSIONS
    # A is n x m, B is m x p, C is n x p
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    target_rank = 46
    
    # Vectorized ALS
    def run_als(U, V, W, max_iters, time_limit):
        U_3d = U.reshape(n, m, target_rank)
        V_3d = V.reshape(m, p, target_rank)
        W_3d = W.reshape(p, n, target_rank)
        
        best_err = np.inf
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = V_3d[j] * W_3d[:, i]  # (p, R)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U_3d[i, j] = sol
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = U_3d[:, j] * W_3d[k]  # (n, R)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V_3d[j, k] = sol
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = U_3d[i] * V_3d[:, k]  # (m, R)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W_3d[k, i] = sol
                    except:
                        pass
            
            # Check error
            if it % 5 == 0:
                T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                err = np.sum((T - 1.0)**2)
                if err < best_err:
                    best_err = err
                if err < 1e-10:
                    break
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.sum((T - 1.0)**2)
        return U_3d.reshape(n*m, target_rank), V_3d.reshape(m*p, target_rank), W_3d.reshape(p*n, target_rank), err

    # Multiple restarts
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        scale = 1.0 / np.sqrt(target_rank)
        U = np.random.randn(n * m, target_rank) * scale
        V = np.random.randn(m * p, target_rank) * scale
        W = np.random.randn(p * n, target_rank) * scale
        
        U_res, V_res, W_res, err = run_als(U, V, W, 500, deadline - 2)
        
        if err < 1e-8:
            best_U, best_V, best_W = U_res, V_res, W_res
            best_rank = target_rank
            break
        elif err < 1e-4:
            # Keep the best found so far if it's better than schoolbook?
            # Actually we only care about rank. If we don't find 46, we return 54.
            # But maybe we can try to polish?
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
