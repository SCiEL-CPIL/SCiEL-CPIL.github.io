import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor M
    M = np.zeros((n*m, m*p, p*n))
    nz_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i*m + j
                idx2 = j*p + k
                idx3 = k*n + i
                M[idx1, idx2, idx3] = 1.0
                nz_idx.append((idx1, idx2, idx3))
    nz_idx = np.array(nz_idx)
    
    # Precompute flattened views for ALS
    M_flat = M.reshape(n*m, m*p * p*n)
    M_T_flat = M.T.reshape(m*p, n*m*p*n)
    M_P_flat = M.reshape(n*m*m*p, p*n).T
    
    # Precompute Schoolbook factors for "Span" initialization
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
                
    best_result = None
    best_rank = None
    
    lower_bound = 24
    
    for rank in range(lower_bound, 73):
        if time.time() > deadline - 20:
            break
        
        found = False
        
        if rank <= 26:
            max_restarts = 40
        elif rank <= 30:
            max_restarts = 25
        else:
            max_restarts = 10
            
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            seed = rank * 1000 + restart
            np.random.seed(seed)
            
            if restart % 3 == 0:
                # Random Gaussian
                U = np.random.randn(n*m, rank)
                V = np.random.randn(m*p, rank)
                W = np.random.randn(p*n, rank)
            elif restart % 3 == 1:
                # Schoolbook Span Initialization
                A = np.random.randn(rank_sb, rank)
                B = np.random.randn(rank_sb, rank)
                C = np.random.randn(rank_sb, rank)
                U = U_sb @ A
                V = V_sb @ B
                W = W_sb @ C
            else:
                # Sparse Random
                U = np.random.randn(n*m, rank)
                V = np.random.randn(m*p, rank)
                W = np.random.randn(p*n, rank)
                mask_u = np.random.rand(n*m, rank) < 0.2
                mask_v = np.random.rand(m*p, rank) < 0.2
                mask_w = np.random.rand(p*n, rank) < 0.2
                U *= mask_u
                V *= mask_v
                W *= mask_w
                if np.all(U == 0): U[0,0] = 1
                if np.all(V == 0): V[0,0] = 1
                if np.all(W == 0): W[0,0] = 1
            
            converged = False
            prev_residual = np.inf
            min_residual = np.inf
            
            for iteration in range(100):
                if time.time() > deadline - 10:
                    break
                
                T = (V[:, np.newaxis, :] * W[np.newaxis, :, :])
                T_flat = T.reshape(m*p*p*n, rank)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(T_flat.T, M_flat.T, rcond=None)
                    U_new = U_new.T
                except:
                    break
                
                T2 = (U[:, np.newaxis, :] * W[np.newaxis, :, :])
                T2_flat = T2.reshape(n*m*p*n, rank)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(T2_flat.T, M_T_flat.T, rcond=None)
                    V_new = V_new.T
                except:
                    break
                
                T3 = (U[:, np.newaxis, :] * V[np.newaxis, :, :])
                T3_flat = T3.reshape(n*m*m*p, rank)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(T3_flat.T, M_P_flat.T, rcond=None)
                    W_new = W_new.T
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                i_vals = nz_idx[:, 0]
                j_vals = nz_idx[:, 1]
                k_vals = nz_idx[:, 2]
                U_vals = U[i_vals, :]
                V_vals = V[j_vals, :]
                W_vals = W[k_vals, :]
                tensor_vals = np.sum(U_vals * V_vals * W_vals, axis=1)
                residual = np.abs(tensor_vals - 1.0).max()
                
                if residual < 1e-10:
                    converged = True
                    break
                
                if residual < min_residual:
                    min_residual = residual
                
                if residual > prev_residual * 0.999 and iteration > 10:
                    if iteration > 30:
                        break
                prev_residual = residual
            
            # Polish Phase with SciPy Minimize if ALS got close but not exact
            if not converged and min_residual < 1e-3:
                if time.time() > deadline - 15:
                    break
                
                def objective(x):
                    U_r = x[:n*m*rank].reshape(n*m, rank)
                    V_r = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_r = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    U_vals = U_r[i_vals, :]
                    V_vals = V_r[j_vals, :]
                    W_vals = W_r[k_vals, :]
                    tensor_vals = np.sum(U_vals * V_vals * W_vals, axis=1)
                    diff = tensor_vals - 1.0
                    return np.sum(diff * diff)
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                try:
                    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12})
                    if res.success or res.fun < 1e-10:
                        U_pol = res.x[:n*m*rank].reshape(n*m, rank)
                        V_pol = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                        W_pol = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                        
                        U_vals = U_pol[i_vals, :]
                        V_vals = V_pol[j_vals, :]
                        W_vals = W_pol[k_vals, :]
                        tensor_vals = np.sum(U_vals * V_vals * W_vals, axis=1)
                        residual = np.abs(tensor_vals - 1.0).max()
                        
                        if residual < 1e-10:
                            converged = True
                            U, V, W = U_pol, V_pol, W_pol
                except Exception:
                    pass
            
            if converged:
                T_check = (U[:, np.newaxis, np.newaxis, :] * V[np.newaxis, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :])
                final_residual = np.abs(np.sum(T_check, axis=3) - M).max()
                
                if final_residual < 1e-10:
                    best_rank = rank
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    else:
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
