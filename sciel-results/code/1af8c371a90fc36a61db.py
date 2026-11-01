import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    
    # Precompute index arrays for vectorized construction
    i_arr = np.repeat(np.arange(n), m * m * p * p * n)
    j_arr = np.tile(np.repeat(np.arange(m), m * p * p * n), n)
    jp_arr = np.tile(np.repeat(np.arange(m), p * p * n), n * m)
    k_arr = np.tile(np.repeat(np.arange(p), p * n), n * m * m)
    kp_arr = np.tile(np.repeat(np.arange(p), n), n * m * m * p)
    ip_arr = np.tile(np.arange(n), n * m * m * p * p)
    
    num_eq = n * m * m * p * p * n
    
    for rank in [35, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]:
        if time.time() - start_time > 100:
            break
        
        for trial in range(3):
            if time.time() - start_time > 115:
                break
            
            np.random.seed(42 + trial * 7)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            
            for iteration in range(30):
                if time.time() - start_time > 118:
                    break
                
                # Build A_w, b_w for solving W
                # A_w[eq, (kp*n+ip)*rank + r] = U[i*m+j, r] * V[jp*p+k, r]
                # b_w[eq] = 1 if j==jp and i==ip and k==kp else 0
                
                b_w = (j_arr == jp_arr) & (i_arr == ip_arr) & (k_arr == kp_arr)
                b_w = b_w.astype(float)
                
                # For each r, build the contribution
                A_w = np.zeros((num_eq, p * n * rank))
                for r in range(rank):
                    u_col = U[:, r]  # shape (n*m,)
                    v_col = V[:, r]  # shape (m*p,)
                    
                    # For each equation, A_w[eq, (kp*n+ip)*rank + r] = u_col[i*m+j] * v_col[jp*p+k]
                    # i*m+j is an index into u_col
                    u_idx = i_arr * m + j_arr  # shape (num_eq,)
                    v_idx = jp_arr * p + k_arr  # shape (num_eq,)
                    
                    vals = u_col[u_idx] * v_col[v_idx]  # shape (num_eq,)
                    
                    # Column index in A_w
                    col_idx = (kp_arr * n + ip_arr) * rank + r  # shape (num_eq,)
                    A_w[np.arange(num_eq), col_idx] = vals
                
                W_flat, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = W_flat.reshape(p * n, rank)
                
                # Build A_v, b_v for solving V
                b_v = b_w.copy()
                A_v = np.zeros((num_eq, m * p * rank))
                for r in range(rank):
                    u_col = U[:, r]
                    w_col = W[:, r]
                    
                    u_idx = i_arr * m + j_arr
                    w_idx = kp_arr * n + ip_arr
                    
                    vals = u_col[u_idx] * w_col[w_idx]
                    
                    col_idx = (jp_arr * p + k_arr) * rank + r
                    A_v[np.arange(num_eq), col_idx] = vals
                
                V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_flat.reshape(m * p, rank)
                
                # Build A_u, b_u for solving U
                b_u = b_w.copy()
                A_u = np.zeros((num_eq, n * m * rank))
                for r in range(rank):
                    v_col = V[:, r]
                    w_col = W[:, r]
                    
                    v_idx = jp_arr * p + k_arr
                    w_idx = kp_arr * n + ip_arr
                    
                    vals = v_col[v_idx] * w_col[w_idx]
                    
                    col_idx = (i_arr * m + j_arr) * rank + r
                    A_u[np.arange(num_eq), col_idx] = vals
                
                U_flat, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_flat.reshape(n * m, rank)
                
                # Check error
                error = 0.0
                for r in range(rank):
                    u_col = U[:, r]
                    v_col = V[:, r]
                    w_col = W[:, r]
                    
                    u_idx = i_arr * m + j_arr
                    v_idx = jp_arr * p + k_arr
                    w_idx = kp_arr * n + ip_arr
                    
                    prod = u_col[u_idx] * v_col[v_idx] * w_col[w_idx]
                    error += np.sum((prod - b_w) ** 2)
                
                if error < 1e-6:
                    U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n * m)]
                    V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m * p)]
                    W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p * n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
