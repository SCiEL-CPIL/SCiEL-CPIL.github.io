import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_eq = n * m * p  # 64
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Precompute index arrays
    eq_indices = np.arange(total_eq)
    i_idx = (eq_indices // (m * p)).astype(int)
    j_idx = ((eq_indices // p) % m).astype(int)
    k_idx = (eq_indices % p).astype(int)
    
    u_row_idx = i_idx * m + j_idx
    v_row_idx = j_idx * p + k_idx
    w_row_idx = k_idx * n + i_idx
    
    def snap_to_half_int(X):
        return np.round(2 * X) / 2.0
    
    def compute_residual_snapped(U, V, W):
        U_s = snap_to_half_int(U)
        V_s = snap_to_half_int(V)
        W_s = snap_to_half_int(W)
        U_eq = U_s[u_row_idx, :]
        V_eq = V_s[v_row_idx, :]
        W_eq = W_s[w_row_idx, :]
        vals = np.sum(U_eq * V_eq * W_eq, axis=1)
        return np.sum((vals - 1)**2)
    
    def solve_for_U(V, W, target_rank):
        A_mat = np.zeros((total_eq, nm * target_rank))
        for eq in range(total_eq):
            u_r = u_row_idx[eq]
            v_r = v_row_idx[eq]
            w_r = w_row_idx[eq]
            coeffs = V[v_r, :] * W[w_r, :]
            A_mat[eq, u_r*target_rank : (u_r+1)*target_rank] = coeffs
        b_vec = np.ones(total_eq)
        U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        return U_flat.reshape(nm, target_rank)
    
    def solve_for_V(U, W, target_rank):
        A_mat = np.zeros((total_eq, mp * target_rank))
        for eq in range(total_eq):
            v_r = v_row_idx[eq]
            u_r = u_row_idx[eq]
            w_r = w_row_idx[eq]
            coeffs = U[u_r, :] * W[w_r, :]
            A_mat[eq, v_r*target_rank : (v_r+1)*target_rank] = coeffs
        b_vec = np.ones(total_eq)
        V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        return V_flat.reshape(mp, target_rank)
    
    def solve_for_W(U, V, target_rank):
        A_mat = np.zeros((total_eq, pn * target_rank))
        for eq in range(total_eq):
            w_r = w_row_idx[eq]
            u_r = u_row_idx[eq]
            v_r = v_row_idx[eq]
            coeffs = U[u_r, :] * V[v_r, :]
            A_mat[eq, w_r*target_rank : (w_r+1)*target_rank] = coeffs
        b_vec = np.ones(total_eq)
        W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        return W_flat.reshape(pn, target_rank)
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    for target_rank in range(32, 64):
        if time.time() > deadline:
            break
        
        max_iters = 20
        for init in range(max_iters):
            if time.time() > deadline:
                break
            
            np.random.seed(init * 1000 + target_rank)
            
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            prev_residual = float('inf')
            for it in range(200):
                if time.time() > deadline:
                    break
                
                U = solve_for_U(V, W, target_rank)
                V = solve_for_V(U, W, target_rank)
                W = solve_for_W(U, V, target_rank)
                
                U_eq = U[u_row_idx, :]
                V_eq = V[v_row_idx, :]
                W_eq = W[w_row_idx, :]
                vals = np.sum(U_eq * V_eq * W_eq, axis=1)
                residual = np.sum((vals - 1)**2)
                
                if residual < 1e-8:
                    snapped_res = compute_residual_snapped(U, V, W)
                    if snapped_res < 1e-10:
                        U_s = snap_to_half_int(U)
                        V_s = snap_to_half_int(V)
                        W_s = snap_to_half_int(W)
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
                if abs(prev_residual - residual) < 1e-15:
                    break
                prev_residual = residual
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
