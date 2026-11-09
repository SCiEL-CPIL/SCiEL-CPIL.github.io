import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # 1. Generate the Schoolbook Decomposition (rank 45)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = get_schoolbook()
    
    # Verify schoolbook decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_sb[i*m+j, :] * V_sb[j*p+k, :] * W_sb[k*n+i, :])
                if not np.isclose(s, 1.0):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    best_result = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    best_rank = n * m * p
    
    # Try to find lower rank decompositions via Alternating Least Squares (ALS)
    def try_rank_als(r, max_iters=200, restarts=10):
        if time.time() - start_time > deadline - 10:
            return None
        
        for restart in range(restarts):
            if time.time() - start_time > deadline - 5:
                break
            
            # Initialize
            np.random.seed(42 + restart * 100)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Scale to avoid under/overflow
            norm_u = np.linalg.norm(U)
            norm_v = np.linalg.norm(V)
            norm_w = np.linalg.norm(W)
            if norm_u > 0: U /= norm_u
            if norm_v > 0: V /= norm_v
            if norm_w > 0: W /= norm_w
            
            for it in range(max_iters):
                # Update U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = u_sol
                        except Exception:
                            U_new[i*m+j, :] = U[i*m+j, :]
                U = U_new
                
                # Update V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, r))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = v_sol
                        except Exception:
                            V_new[j*p+k, :] = V[j*p+k, :]
                V = V_new
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, r))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = w_sol
                        except Exception:
                            W_new[k*n+i, :] = W[k*n+i, :]
                W = W_new
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = abs(s - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-10:
                    # Round and verify
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                                if not np.isclose(s, 1.0, atol=1e-9):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        return (U_r, V_r, W_r)
                    # If rounding failed, break to try next restart
                    break

        return None
    
    # Try ranks from 44 down to 20
    for r in range(44, 19, -1):
        if time.time() - start_time > deadline - 20:
            break
        
        result = try_rank_als(r)
        if result is not None:
            U, V, W = result
            best_result = (U.copy(), V.copy(), W.copy())
            best_rank = r
            break
    
    U_out, V_out, W_out = best_result
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
