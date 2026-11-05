import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Dimensions
    dim_u = n * m  # 9
    dim_v = m * p  # 24
    dim_w = p * n  # 24
    
    # Target Tensor
    # T[a, b, c] = 1 if indices match multiplication
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.reshape(dim_u, -1) # (9, 576)
    
    def verify(U, V, W):
        # Check exactness
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        # return np.allclose(T_recon, T, atol=1e-9)
        # Faster check using the flattened form
        # T_recon_flat[a, b*c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can just compute the error norm
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_recon, T, atol=1e-8, rtol=1e-8)

    def als_step(U, V, W, rank):
        """Perform one ALS iteration. Returns updated U, V, W."""
        # Update U
        # For each a, solve M_a @ u_a = t_a
        # M_a[idx, r] = V[b, r] * W[c, r] where idx = b * dim_w + c
        M_U = np.zeros((dim_v * dim_w, rank))
        for b in range(dim_v):
            for c in range(dim_w):
                idx = b * dim_w + c
                M_U[idx, :] = V[b, :] * W[c, :]
        
        U_new = np.zeros_like(U)
        for a in range(dim_u):
            # M_U @ u_a = T_flat[a]
            # u_a is (rank, 1)
            U_new[a, :] = np.linalg.lstsq(M_U, T_flat[a], rcond=None)[0]
        
        # Update V
        # For each b, solve M_b @ v_b = t_b
        # M_b[idx, r] = U[a, r] * W[c, r] where idx = a * dim_w + c
        M_V = np.zeros((dim_u * dim_w, rank))
        for a in range(dim_u):
            for c in range(dim_w):
                idx = a * dim_w + c
                M_V[idx, :] = U_new[a, :] * W[c, :]
                
        V_new = np.zeros_like(V)
        for b in range(dim_v):
            # t_b is T[:, b, :].flatten()
            t_b = T[:, b, :].flatten()
            V_new[b, :] = np.linalg.lstsq(M_V, t_b, rcond=None)[0]
            
        # Update W
        # For each c, solve M_c @ w_c = t_c
        # M_c[idx, r] = U[a, r] * V[b, r] where idx = a * dim_v + b
        M_W = np.zeros((dim_u * dim_v, rank))
        for a in range(dim_u):
            for b in range(dim_v):
                idx = a * dim_v + b
                M_W[idx, :] = U_new[a, :] * V_new[b, :]
                
        W_new = np.zeros_like(W)
        for c in range(dim_w):
            # t_c is T[:, :, c].flatten()
            t_c = T[:, :, c].flatten()
            W_new[c, :] = np.linalg.lstsq(M_W, t_c, rcond=None)[0]
            
        return U_new, V_new, W_new

    # Try ranks from lower bound to schoolbook
    # Lower bound is max(9, 24, 24) = 24
    # Schoolbook is 72
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = 72
    
    # Initialize schoolbook
    U_school = np.zeros((dim_u, 72))
    V_school = np.zeros((dim_v, 72))
    W_school = np.zeros((dim_w, 72))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school

    for target_rank in range(25, 72):
        if time.time() > deadline - 20:
            break
            
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.RandomState(42 + attempt + target_rank)
            U = rng.randn(dim_u, target_rank) * 0.1
            V = rng.randn(dim_v, target_rank) * 0.1
            W = rng.randn(dim_w, target_rank) * 0.1
            
            # Run ALS
            converged = False
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                    
                U, V, W = als_step(U, V, W, target_rank)
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                err = np.linalg.norm(T_recon - T)
                
                if err < 1e-8:
                    if verify(U, V, W):
                        best_U, best_V, best_W = U, V, W
                        best_rank = target_rank
                        converged = True
                        break
                        
            if converged:
                break
        
        if best_rank < 72:
            break
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
