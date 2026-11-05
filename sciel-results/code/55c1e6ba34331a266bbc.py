import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140

def construct():
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    # Build the target tensor T[a,b,c] where:
    # a = i*m + j (i in [0,n), j in [0,m))
    # b = j*p + k (j in [0,m), k in [0,p))
    # c = k*n + i (k in [0,p), i in [0,n))
    # T[a,b,c] = 1 if the indices are consistent, else 0
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Start with the known rank-66 decomposition
    rank_init = 66
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank_init or not check_deadline():
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx, current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, current_rank] += val
        current_rank += 1
        
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
                
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # Strassen 7
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
        
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
    
    # Now try to reduce rank using ALS with hard thresholding
    def verify_decomposition(U_mat, V_mat, W_mat, r):
        """Verify that the decomposition is exact."""
        if r == 0:
            return False
        T_recon = np.einsum('ar,br,cr->abc', U_mat[:, :r], V_mat[:, :r], W_mat[:, :r])
        return np.allclose(T, T_recon, atol=1e-8)
    
    def als_reduce(U_init, V_init, W_init, r_init, target_rank, max_iter=200, tol=1e-6):
        """ALS with hard thresholding to reduce rank."""
        if not check_deadline():
            return U_init, V_init, W_init, r_init
        
        best_U = U_init[:, :r_init].copy()
        best_V = V_init[:, :r_init].copy()
        best_W = W_init[:, :r_init].copy()
        best_r = r_init
        best_err = np.inf
        
        # Try different target ranks
        for target_r in range(max(1, target_rank), min(r_init, 60) + 1):
            if not check_deadline():
                break
            
            # Initialize with first target_r components
            U_cur = U_init[:, :target_r].copy()
            V_cur = V_init[:, :target_r].copy()
            W_cur = W_init[:, :target_r].copy()
            
            prev_err = np.inf
            for it in range(max_iter):
                if not check_deadline():
                    break
                
                # Update U: for each a, U[a, :] = (V^T * W^T * T[a, :, :]) / (V * W^T)
                # More efficiently: U[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] / sum_r' V[b,r'] * W[c,r']
                # But this is complex. Use gradient-based approach instead.
                
                # Compute error
                T_recon = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
                err = np.linalg.norm(T - T_recon, 'fro')
                
                if err < best_err:
                    best_err = err
                    best_U = U_cur.copy()
                    best_V = V_cur.copy()
                    best_W = W_cur.copy()
                    best_r = target_r
                
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
                
                # Gradient descent step
                lr = 0.01
                # Gradient w.r.t. U
                grad_U = np.einsum('abc,br,cr->ar', T - T_recon, V_cur, W_cur)
                grad_V = np.einsum('abc,ar,cr->br', T - T_recon, U_cur, W_cur)
                grad_W = np.einsum('abc,ar,br->cr', T - T_recon, U_cur, V_cur)
                
                U_cur -= lr * grad_U
                V_cur -= lr * grad_V
                W_cur -= lr * grad_W
                
                # Hard thresholding: zero out small entries
                U_cur[np.abs(U_cur) < 1e-3] = 0
                V_cur[np.abs(V_cur) < 1e-3] = 0
                W_cur[np.abs(W_cur) < 1e-3] = 0
                
                # Try to snap to integers or half-integers
                U_cur = np.round(U_cur * 2) / 2
                V_cur = np.round(V_cur * 2) / 2
                W_cur = np.round(W_cur * 2) / 2
                
                # Check if we can verify
                if verify_decomposition(U_cur, V_cur, W_cur, target_r):
                    return U_cur, V_cur, W_cur, target_r
            
            if best_r <= target_r:
                break
        
        return best_U, best_V, best_W, best_r
    
    # Try ALS reduction
    if check_deadline():
        U_red, V_red, W_red, reduced_rank = als_reduce(U, V, W, current_rank, 50, max_iter=100)
        if reduced_rank < current_rank and verify_decomposition(U_red, V_red, W_red, reduced_rank):
            U = U_red
            V = V_red
            W = W_red
            current_rank = reduced_rank
    
    # Final verification
    if not verify_decomposition(U, V, W, current_rank):
        # Fall back to original
        U = np.zeros((n * m, rank_init), dtype=float)
        V = np.zeros((m * p, rank_init), dtype=float)
        W = np.zeros((p * n, rank_init), dtype=float)
        current_rank = 0
        for block_k in range(3):
            k_off = block_k * 2
            add_1_4_2(0, 0, k_off)
            add_2_4_2(1, 0, k_off)
    
    return {"U": U[:, :current_rank].tolist(), "V": V[:, :current_rank].tolist(), "W": W[:, :current_rank].tolist()}
