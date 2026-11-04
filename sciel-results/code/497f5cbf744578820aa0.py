import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_fb, V_fb, W_fb, best_rank = schoolbook()
    
    # Precompute the non-zero indices of the target tensor T
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + j  # j' = j, k' = j
                w = k * n + i  # i' = i
                nz_indices.append((u, v, w))
                
    def check_solution(U_mat, V_mat, W_mat):
        T_calc = np.einsum('ur,vr,wr->uvw', U_mat, V_mat, W_mat)
        T_target = np.zeros((n*m, m*p, p*n))
        for u, v, w in nz_indices:
            T_target[u, v, w] = 1.0
        if np.allclose(T_calc, T_target, atol=1e-6):
            return True
        return False

    def to_half_int(mat):
        return np.round(mat * 2) / 2

    def run_als(rank, max_iters=50, restarts=5):
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                # Update W
                U_exp = U[:, np.newaxis, :]
                V_exp = V[np.newaxis, :, :]
                A = (U_exp * V_exp).reshape(n*m * m*p, rank)
                
                T_mat = np.zeros((n*m * m*p, p*n))
                for u, v, w in nz_indices:
                    T_mat[u * m*p + v, w] = 1.0
                    
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A, T_mat, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                # Update U
                V_exp = V[:, np.newaxis, :]
                W_exp = W[np.newaxis, :, :]
                B = (V_exp * W_exp).reshape(m*p * p*n, rank)
                
                T_mat_U = np.zeros((n*m, m*p * p*n))
                for u, v, w in nz_indices:
                    T_mat_U[u, v * p*n + w] = 1.0
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(B, T_mat_U, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                U_exp = U[:, np.newaxis, :]
                W_exp = W[np.newaxis, :, :]
                C = (U_exp * W_exp).reshape(n*m * p*n, rank)
                
                T_mat_V = np.zeros((m*p, n*m * p*n))
                for u, v, w in nz_indices:
                    T_mat_V[v, u * p*n + w] = 1.0
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(C, T_mat_V, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                T_calc = np.einsum('ur,vr,wr->uvw', U, V, W)
                T_target = np.zeros((n*m, m*p, p*n))
                for u, v, w in nz_indices:
                    T_target[u, v, w] = 1.0
                
                err = np.linalg.norm(T_calc - T_target)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if err < 1e-5:
                    break
                    
        if best_err < 1e-5:
            U_h = to_half_int(best_U)
            V_h = to_half_int(best_V)
            W_h = to_half_int(best_W)
            if check_solution(U_h, V_h, W_h):
                return U_h, V_h, W_h
        return None

    lower_bound = 27
    found = False
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 20:
            break
            
        result = run_als(rank, max_iters=20, restarts=10)
        if result is not None:
            U, V, W = result
            found = True
            best_rank = rank
            break
            
    if not found:
        U, V, W = U_fb, V_fb, W_fb

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
