import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    np.random.seed(42)
    random.seed(42)

    def check_validity_residual(U, V, W, tol=1e-8):
        R = U.shape[1]
        V_mat = V.reshape(m, p, R)
        U_mat = U.reshape(n, m, R)
        W_mat = W.reshape(p, n, R)
        
        err = 0.0
        for i in range(n):
            if time.time() > deadline:
                return float('inf')
            for k in range(p):
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                W_k_i = W_mat[k, i, :]
                M_ik = U_i @ np.diag(W_k_i) @ V_k.T
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    def get_rank_55_solution():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            U[0*m+0, col] += 1
            U[1*m+1, col] += 1
            V[0*p+k0, col] += 1
            V[1*p+k1, col] += 1
            W[k0*n+0, col] += 1
            W[k1*n+1, col] += 1
            col += 1
            
            U[1*m+0, col] += 1
            U[1*m+1, col] += 1
            V[0*p+k0, col] += 1
            W[k0*n+1, col] += 1
            W[k1*n+1, col] += -1
            col += 1
            
            U[0*m+0, col] += 1
            V[0*p+k1, col] += 1
            V[1*p+k1, col] += -1
            W[k1*n+0, col] += 1
            W[k1*n+1, col] += 1
            col += 1
            
            U[1*m+1, col] += 1
            V[1*p+k0, col] += 1
            V[0*p+k0, col] += -1
            W[k0*n+0, col] += 1
            W[k0*n+1, col] += 1
            col += 1
            
            U[0*m+0, col] += 1
            U[0*m+1, col] += 1
            V[1*p+k1, col] += 1
            W[k0*n+0, col] += -1
            W[k1*n+0, col] += 1
            col += 1
            
            U[1*m+0, col] += 1
            U[0*m+0, col] += -1
            V[0*p+k0, col] += 1
            V[0*p+k1, col] += 1
            W[k1*n+1, col] += 1
            col += 1
            
            U[0*m+1, col] += 1
            U[1*m+1, col] += -1
            V[1*p+k0, col] += 1
            V[1*p+k1, col] += 1
            W[k0*n+0, col] += 1
            col += 1
        
        return U, V, W

    U_55, V_55, W_55 = get_rank_55_solution()
    
    u_rows = [0, 1, 3, 4]
    v_rows = list(range(20))
    w_rows = list(range(20))
    
    U_sub = U_55[u_rows, 20:]
    V_sub = V_55[v_rows, 20:]
    W_sub = W_55[w_rows, 20:]
    
    def check_2x2x10(U, V, W):
        R = U.shape[1]
        U_mat = U.reshape(2, 2, R)
        V_mat = V.reshape(2, 10, R)
        W_mat = W.reshape(10, 2, R)
        
        err = 0.0
        for i in range(2):
            for k in range(10):
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                W_k_i = W_mat[k, i, :]
                M_ik = U_i @ np.diag(W_k_i) @ V_k.T
                diff = M_ik - np.eye(2)
                err += np.sum(diff**2)
        return err

    best_U_sub, best_V_sub, best_W_sub = U_sub, V_sub, W_sub
    best_rank_sub = 35
    
    def try_reduce_2x2x10(U, V, W, R, max_time=80.0):
        start = time.time()
        U, V, W = U.copy(), V.copy(), W.copy()
        
        while time.time() - start < max_time:
            reduced = False
            
            for c in range(R):
                if time.time() - start > max_time:
                    return U, V, W, R
                U_r = np.delete(U, c, axis=1)
                V_r = np.delete(V, c, axis=1)
                W_r = np.delete(W, c, axis=1)
                err = check_2x2x10(U_r, V_r, W_r)
                if err < 1e-6:
                    U, V, W = U_r, V_r, W_r
                    R -= 1
                    reduced = True
                    break
            
            if reduced:
                continue
            
            if R > 28:
                for attempt in range(20):
                    if time.time() - start > max_time:
                        return U, V, W, R
                    
                    U_p = U + 0.03 * np.random.randn(U.shape[0], R)
                    V_p = V + 0.03 * np.random.randn(V.shape[0], R)
                    
                    W_p = np.zeros((20, R))
                    valid = True
                    for k in range(10):
                        for i in range(2):
                            A_mat = np.zeros((4, R))
                            b_vec = np.zeros(4)
                            for j in range(2):
                                for jp in range(2):
                                    idx = j * 2 + jp
                                    A_mat[idx, :] = U_p[i*2+j, :] * V_p[jp*10+k, :]
                                    if j == jp:
                                        b_vec[idx] = 1
                            try:
                                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                                W_p[k*2+i, :] = sol
                            except:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        err = check_2x2x10(U_p, V_p, W_p)
                        if err < 1e-6:
                            U, V, W = U_p, V_p, W_p
                            for c in range(R):
                                if time.time() - start > max_time:
                                    return U, V, W, R
                                U_r = np.delete(U, c, axis=1)
                                V_r = np.delete(V, c, axis=1)
                                W_r = np.delete(W, c, axis=1)
                                err_r = check_2x2x10(U_r, V_r, W_r)
                                if err_r < 1e-6:
                                    U, V, W = U_r, V_r, W_r
                                    R -= 1
                                    break
                            break
        
        return U, V, W, R

    if time.time() < deadline - 30:
        best_U_sub, best_V_sub, best_W_sub, best_rank_sub = try_reduce_2x2x10(
            U_sub, V_sub, W_sub, 35, max_time=80.0
        )
    
    if best_rank_sub < 35:
        total_rank = 20 + best_rank_sub
        U_new = np.zeros((6, total_rank))
        V_new = np.zeros((30, total_rank))
        W_new = np.zeros((20, total_rank))
        
        U_new[:, :20] = U_55[:, :20]
        V_new[:, :20] = V_55[:, :20]
        W_new[:, :20] = W_55[:, :20]
        
        U_new[u_rows, 20:] = best_U_sub
        V_new[v_rows, 20:] = best_V_sub
        W_new[w_rows, 20:] = best_W_sub
        
        err_final = check_validity_residual(U_new, V_new, W_new)
        if err_final < 1e-6:
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    return {"U": U_55.tolist(), "V": V_55.tolist(), "W": W_55.tolist()}
