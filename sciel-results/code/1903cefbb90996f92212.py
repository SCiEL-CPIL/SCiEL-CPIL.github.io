import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Target: 2x3x6
    # Strategy: Find rank-9 decomposition for 2x3x3 block, replicate for 2 blocks -> Rank 18.
    # Or find rank-18 for full tensor directly?
    # The decoupled ALS is fast enough to try rank 9 on the block.
    
    # Block T for 2x3x3
    n_b, m_b, p_b = 2, 3, 3
    T_block = np.zeros((n_b * m_b, m_b * p_b, p_b * n_b))
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                a = i * m_b + j
                b = j * p_b + k
                c = i * p_b + k
                T_block[a, b, c] = 1.0
                
    # Dimensions for block
    dim_a = n_b * m_b # 6
    dim_b = m_b * p_b # 9
    dim_c = p_b * n_b # 6
    
    rank_target = 9
    max_iters = 100
    num_restarts = 20 # Can afford more with fast solver
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    found = False
    
    # Pre-compute index mappings for the decoupled systems
    # For U (fixed a): rows are (b, c) -> b*dim_c + c
    # For V (fixed b): rows are (a, c) -> a*dim_c + c
    # For W (fixed c): rows are (a, b) -> a*dim_b + b
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        U = np.random.randn(dim_a, rank_target)
        V = np.random.randn(dim_b, rank_target)
        W = np.random.randn(dim_c, rank_target)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            # Update U
            # For each a in 0..dim_a-1
            for a in range(dim_a):
                # A_u[a] is dim_b*dim_c x rank_target
                # A_u[a][b*dim_c + c, r] = V[b, r] * W[c, r]
                # Vectorized construction:
                # V is dim_b x R, W is dim_c x R
                # We want matrix of shape (dim_b, dim_c, R) -> V[b, :, None] * W[c, None, :]
                # Reshape to (dim_b * dim_c, R)
                A_u_a = (V[:, :, None] * W[None, :, :]).reshape(-1, rank_target)
                t_u_a = T_block[a, :, :].reshape(-1)
                
                # Solve A_u_a * u_vec = t_u_a
                try:
                    u_vec, _, _, _ = np.linalg.lstsq(A_u_a, t_u_a, rcond=None)
                except:
                    break
                U[a, :] = u_vec
                
            # Update V
            for b in range(dim_b):
                # A_v[b] is dim_a*dim_c x rank_target
                # A_v[b][a*dim_c + c, r] = U[a, r] * W[c, r]
                A_v_b = (U[:, :, None] * W[None, :, :]).reshape(-1, rank_target)
                t_v_b = T_block[:, b, :].reshape(-1)
                
                try:
                    v_vec, _, _, _ = np.linalg.lstsq(A_v_b, t_v_b, rcond=None)
                except:
                    break
                V[b, :] = v_vec
                
            # Update W
            for c in range(dim_c):
                # A_w[c] is dim_a*dim_b x rank_target
                # A_w[c][a*dim_b + b, r] = U[a, r] * V[b, r]
                A_w_c = (U[:, :, None] * V[None, :, :]).reshape(-1, rank_target)
                t_w_c = T_block[:, :, c].reshape(-1)
                
                try:
                    w_vec, _, _, _ = np.linalg.lstsq(A_w_c, t_w_c, rcond=None)
                except:
                    break
                W[c, :] = w_vec
                
            # Check error
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_rec - T_block)
            
            if err < 1e-5:
                found = True
                break
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
        if found:
            break

    # If found, construct full rank-18 solution
    if found and best_err < 1e-5:
        U_full = np.zeros((dim_a, 18))
        V_full = np.zeros((m * p, 18)) # 18 x 18
        W_full = np.zeros((p * n, 18)) # 12 x 18
        
        # Block 1 (B1)
        U_full[:, 0:9] = best_U
        V_full[0:9, 0:9] = best_V
        W_full[0:6, 0:9] = best_W
        
        # Block 2 (B2)
        U_full[:, 9:18] = best_U
        V_full[9:18, 9:18] = best_V
        W_full[6:12, 9:18] = best_W
        
        def round_half(x):
            return np.round(x * 2) / 2
            
        U_r = round_half(U_full)
        V_r = round_half(V_full)
        W_r = round_half(W_full)
        
        # Verify full tensor
        T_full = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = i * p + k
                    T_target[a, b, c] = 1.0
                    
        err_r = np.linalg.norm(T_full - T_target)
        
        if err_r < 1e-5:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Fallback: Schoolbook Rank 36
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
