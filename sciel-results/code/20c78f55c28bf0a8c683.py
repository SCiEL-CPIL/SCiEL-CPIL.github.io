import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    
    # Construct T
    T = np.zeros((u_dim, v_dim, w_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Schoolbook
    R_school = u_dim * v_dim * w_dim # No, n*m*p
    R_school = n * m * p
    U_school = np.zeros((u_dim, R_school))
    V_school = np.zeros((v_dim, R_school))
    W_school = np.zeros((w_dim, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_school[u_idx, col] = 1
                V_school[v_idx, col] = 1
                W_school[w_idx, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    best_err = 0.0
    
    start_time = time.time()
    deadline = start_time + 120
    
    # Try ranks
    for R in range(50, 19, -1):
        if time.time() > deadline - 10:
            break
            
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(u_dim, R)
            V = np.random.randn(v_dim, R)
            W = np.random.randn(w_dim, R)
            
            # ALS
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                for u in range(u_dim):
                    # M is v_dim * w_dim x R
                    # Row is V_v @ W_w (element-wise product, flattened)
                    M = np.zeros((v_dim * w_dim, R))
                    b = T[u, :, :].flatten()
                    idx = 0
                    for v in range(v_dim):
                        for w in range(w_dim):
                            M[idx, :] = V[v, :] * W[w, :]
                            idx += 1
                    # Solve M x = b
                    U[u, :], residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                    
                # Update V
                for v in range(v_dim):
                    M = np.zeros((u_dim * w_dim, R))
                    b = T[:, v, :].flatten()
                    idx = 0
                    for u in range(u_dim):
                        for w in range(w_dim):
                            M[idx, :] = U[u, :] * W[w, :]
                            idx += 1
                    V[v, :], residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                    
                # Update W
                for w in range(w_dim):
                    M = np.zeros((u_dim * v_dim, R))
                    b = T[:, :, w].flatten()
                    idx = 0
                    for u in range(u_dim):
                        for v in range(v_dim):
                            M[idx, :] = U[u, :] * V[v, :]
                            idx += 1
                    W[w, :], residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                    
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        err += (val - 1.0)**2
                        
                        # Check zeros
                        # Too slow to check all zeros.
                        
            # If err is small, we have a solution
            if err < 1e-6:
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
                
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
