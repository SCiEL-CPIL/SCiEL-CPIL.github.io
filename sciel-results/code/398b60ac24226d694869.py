import numpy as np
import time
import random

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    def is_exact(U, V, W, tol=1e-9):
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        diff = np.max(np.abs(T_recon - T))
        return diff < tol

    def round_half_int(M):
        return 0.5 * np.round(2 * M)

    def try_rank(r, max_iters=100, restarts=10, deadline=None):
        for restart in range(restarts):
            if deadline and time.time() > deadline:
                return None
                
            U = np.random.randn(dim_u, r) * 0.1
            V = np.random.randn(dim_v, r) * 0.1
            W = np.random.randn(dim_w, r) * 0.1
            
            for it in range(max_iters):
                if deadline and time.time() > deadline:
                    return None

                T_reshaped = T.reshape(dim_u, dim_v * dim_w)
                M = (V[:, :, None] * W[None, :, :]).reshape(dim_v * dim_w, r)
                U, _, _, _ = np.linalg.lstsq(M.T, T_reshaped.T, rcond=None)
                U = U.T
                
                T_reshaped2 = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                M2 = (U[:, :, None] * W[None, :, :]).reshape(dim_u * dim_w, r)
                V, _, _, _ = np.linalg.lstsq(M2.T, T_reshaped2.T, rcond=None)
                V = V.T
                
                T_reshaped3 = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                M3 = (U[:, :, None] * V[None, :, :]).reshape(dim_u * dim_v, r)
                W, _, _, _ = np.linalg.lstsq(M3.T, T_reshaped3.T, rcond=None)
                W = W.T
                
                if it % 5 == 0:
                    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
                    err = np.max(np.abs(T_recon - T))
                    if err < 1e-4:
                        U_r = round_half_int(U)
                        V_r = round_half_int(V)
                        W_r = round_half_int(W)
                        if is_exact(U_r, V_r, W_r):
                            return U_r, V_r, W_r
        return None

    start_time = time.time()
    deadline = start_time + 140
    
    def schoolbook():
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
        
    # Try ranks
    for r in [34, 35, 36, 37, 38, 40, 42, 44, 48]:
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 15:
            break
            
        max_iters = 100
        restarts = 10
        if remaining < 30:
            max_iters = 50
            restarts = 5
            
        result = try_rank(r, max_iters, restarts, deadline)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if is_exact(np.array(U_list), np.array(V_list), np.array(W_list)):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return schoolbook()
