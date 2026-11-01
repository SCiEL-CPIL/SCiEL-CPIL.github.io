import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    
    # Target: Find a decomposition for 2x2x4 with rank 12
    # Then use it twice for 2x2x8 to get rank 24.
    
    # Define 2x2x4 tensor
    # A: 2x2 (4 elems), B: 2x4 (8 elems), C: 2x4 (8 elems)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    n_sub, m_sub, p_sub = 2, 2, 4
    U_dim = n_sub * m_sub # 4
    V_dim = m_sub * p_sub # 8
    W_dim = p_sub * n_sub # 8
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                u_idx = i * m_sub + j
                v_idx = j * p_sub + k
                w_idx = k * n_sub + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    rank_target = 12
    
    # ALS to find rank-12 decomposition
    # Initialize
    np.random.seed(42)
    U = np.random.randn(U_dim, rank_target)
    V = np.random.randn(V_dim, rank_target)
    W = np.random.randn(W_dim, rank_target)
    
    # Normalize to prevent divergence
    for r in range(rank_target):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 0: U[:, r] /= norm_u
        if norm_v > 0: V[:, r] /= norm_v
        if norm_w > 0: W[:, r] /= norm_w
        
    best_err = float('inf')
    best_U, best_V, best_W = U, V, W
    
    max_iters = 50
    for it in range(max_iters):
        if time.time() > deadline - 10: # Leave time for construction
            break
            
        # Update W
        # T reshaped as (U_dim, V_dim * W_dim)
        # W_r = argmin || T - sum U_r V_r W_r^T ||
        # This is a linear system for each column of W
        # Let M = T matricized to (U_dim, V_dim * W_dim)
        # For each r, we want to minimize || M - sum_{r'} U_{r'} (V_{r'} \otimes W_{r'})^T ||
        # Actually, standard ALS:
        # Fix U, V. Update W.
        # T = sum_r U_r \otimes V_r \otimes W_r
        # Reshape T to (U_dim, V_dim * W_dim)
        T_mat = T.reshape(U_dim, V_dim * W_dim)
        
        # Build matrix A_w of shape (V_dim * W_dim, rank_target)
        # A_w[:, r] = V_r \otimes W_r ? No.
        # The term is U_r (V_r \otimes W_r)^T.
        # So T_mat = U_mat @ A_w.T ?
        # Let's use the mode-3 unfolding.
        # T_(3) is (W_dim, U_dim * V_dim)
        # T_(3) = W @ (V \otimes U).T ?
        
        # Simpler: Use numpy einsum or explicit loops for small sizes.
        # We want to solve for W columns.
        # Minimize || T - sum_r U_r \otimes V_r \otimes W_r ||^2
        # = sum_{u,v,w} (T[u,v,w] - sum_r U[u,r] V[v,r] W[w,r])^2
        
        # Gradient with respect to W[w, r]:
        # -2 sum_{u,v} U[u,r] V[v,r] (T[u,v,w] - sum_{r'} U[u,r'] V[v,r'] W[w,r'])
        # = -2 sum_{u,v} U[u,r] V[v,r] T[u,v,w] + 2 sum_{u,v,r'} U[u,r] V[v,r] U[u,r'] V[v,r'] W[w,r']
        
        # Let's construct the normal equations.
        # A_w[r, r'] = sum_{u,v} U[u,r] V[v,r] U[u,r'] V[v,r']
        # b_w[r, w] = sum_{u,v} U[u,r] V[v,r] T[u,v,w]
        
        A_w = np.zeros((rank_target, rank_target))
        b_w = np.zeros((rank_target, W_dim))
        
        # Precompute UV products
        # UV[r, u, v] = U[u, r] * V[v, r]
        UV = U[:, :, None] * V[None, :, :] # Shape (U_dim, V_dim, rank_target)
        
        for r in range(rank_target):
            for rp in range(rank_target):
                A_w[r, rp] = np.sum(UV[:, :, r] * UV[:, :, rp])
            # b_w[r, :]
            # sum_{u,v} U[u,r] V[v,r] T[u,v,w]
            # T is (U_dim, V_dim, W_dim)
            # We want for each w: sum_{u,v} UV[u,v,r] T[u,v,w]
            # This is a tensor contraction.
            b_w[r, :] = np.einsum('uvr,uvw->w', UV[:, :, r], T)
            
        # Solve A_w W_col = b_w_col for each w?
        # No, A_w is (rank, rank). W is (W_dim, rank).
        # The equation is A_w @ W.T = b_w.T ?
        # Let's check dimensions.
        # A_w[r, rp] W[w, rp] sum_rp = b_w[r, w]
        # So A_w @ W.T = b_w.T
        try:
            W.T = np.linalg.solve(A_w, b_w.T)
        except np.linalg.LinAlgError:
            W.T = np.linalg.lstsq(A_w, b_w.T, rcond=None)[0]
            
        # Update U
        # Similar logic for U.
        # A_u[r, rp] = sum_{v,w} V[v,r] W[w,r] V[v,rp] W[w,rp]
        # b_u[r, u] = sum_{v,w} V[v,r] W[w,r] T[u,v,w]
        
        A_u = np.zeros((rank_target, rank_target))
        b_u = np.zeros((rank_target, U_dim))
        
        # VW[r, v, w] = V[v, r] * W[w, r]
        VW = V[:, :, None] * W[None, :, :] # Shape (V_dim, W_dim, rank_target)
        
        for r in range(rank_target):
            for rp in range(rank_target):
                A_u[r, rp] = np.sum(VW[:, :, r] * VW[:, :, rp])
            b_u[r, :] = np.einsum('vwr,uvw->u', VW[:, :, r], T)
            
        try:
            U.T = np.linalg.solve(A_u, b_u.T)
        except np.linalg.LinAlgError:
            U.T = np.linalg.lstsq(A_u, b_u.T, rcond=None)[0]
            
        # Update V
        A_v = np.zeros((rank_target, rank_target))
        b_v = np.zeros((rank_target, V_dim))
        
        # UW[r, u, w] = U[u, r] * W[w, r]
        UW = U[:, :, None] * W[None, :, :] # Shape (U_dim, W_dim, rank_target)
        
        for r in range(rank_target):
            for rp in range(rank_target):
                A_v[r, rp] = np.sum(UW[:, :, r] * UW[:, :, rp])
            b_v[r, :] = np.einsum('uwr,uvw->v', UW[:, :, r], T)
            
        try:
            V.T = np.linalg.solve(A_v, b_v.T)
        except np.linalg.LinAlgError:
            V.T = np.linalg.lstsq(A_v, b_v.T, rcond=None)[0]
            
        # Calculate error
        # Reconstruct
        Recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        err = np.linalg.norm(T - Recon)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-8:
            break

    # Check if we found a valid decomposition
    if best_err < 1e-8:
        # Use rank 24 solution
        rank = 24
        U_full = np.zeros((4, 24))
        V_full = np.zeros((16, 24))
        W_full = np.zeros((16, 24))
        
        # Block 1: B[0:8], C[0:8] (Indices 0-7 in V and W)
        U_full[:, 0:12] = best_U
        V_full[0:8, 0:12] = best_V
        W_full[0:8, 0:12] = best_W
        
        # Block 2: B[8:16], C[8:16] (Indices 8-15 in V and W)
        U_full[:, 12:24] = best_U
        V_full[8:16, 12:24] = best_V
        W_full[8:16, 12:24] = best_W
        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to Rank 28 Strassen
        # ... (Reuse existing Strassen code)
        U_strassen = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
        ]
        V_strassen = [
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
            [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
        ]
        W_strassen = [
            [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
            [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
        ]
        
        num_blocks = p // 2
        rank = num_blocks * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for t in range(num_blocks):
            if time.time() > deadline:
                break
            for r in range(7):
                for i in range(n * m):
                    U[i][col] = U_strassen[r][i]
                v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
                for l in range(4):
                    V[v_indices[l]][col] = V_strassen[r][l]
                w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
                for l in range(4):
                    W[w_indices[l]][col] = W_strassen[r][l]
                col += 1
                
        return {"U": U, "V": V, "W": W}

