import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_u = n * m  # 4
    dim_v = m * p  # 6
    dim_w = p * n  # 6
    
    # Build the target tensor T of shape (dim_u, dim_v, dim_w)
    # T[u, v, w] = 1 if A_u * B_v contributes to C_w
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Target rank
    target_rank = 10
    # Fallback rank
    fallback_rank = 11
    
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    def try_rank(rank, max_iters=2000, tol=1e-9):
        # Initialize U, V, W
        # Using small random values to avoid symmetry breaking issues
        U = np.random.randn(dim_u, rank) * 0.1
        V = np.random.randn(dim_v, rank) * 0.1
        W = np.random.randn(dim_w, rank) * 0.1
        
        # ALS Iterations
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            # T_u = sum_c U_uc * (V_c * W_c)
            # Reshape T to (dim_u, dim_v * dim_w)
            T_u = T.reshape(dim_u, dim_v * dim_w)
            # Construct M_vw of shape (dim_v * dim_w, rank)
            M_vw = np.einsum('vc,wc->vwc', V, W).reshape(dim_v * dim_w, rank)
            # Solve U M_vw = T_u
            U, _, _, _ = np.linalg.lstsq(M_vw, T_u, rcond=None)
            
            # Update V
            # T_v = sum_c V_vc * (U_c * W_c)
            T_v = np.einsum('uvw->vuw', T).reshape(dim_v, dim_u * dim_w)
            M_uw = np.einsum('uc,wc->ucw', U, W).reshape(dim_u * dim_w, rank)
            V, _, _, _ = np.linalg.lstsq(M_uw, T_v, rcond=None)
            
            # Update W
            # T_w = sum_c W_wc * (U_c * V_c)
            T_w = np.einsum('uvw->wu v', T).reshape(dim_w, dim_u * dim_v)
            M_uv = np.einsum('uc,vc->ucv', U, V).reshape(dim_u * dim_v, rank)
            W, _, _, _ = np.linalg.lstsq(M_uv, T_w, rcond=None)
            
            # Check convergence
            # Compute error
            T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
            err = np.linalg.norm(T - T_recon)
            if err < tol:
                return U, V, W
        
        return None

    def round_and_verify(U, V, W):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Verify
        T_recon = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r)
        if np.allclose(T, T_recon, atol=1e-9):
            return U_r, V_r, W_r
        return None

    # Try rank 10
    best_result = None
    # Multiple restarts to escape local minima
    for restart in range(20):
        if time.time() > deadline:
            break
        res = try_rank(target_rank)
        if res:
            U, V, W = res
            rounded = round_and_verify(U, V, W)
            if rounded:
                best_result = rounded
                break
    
    # If rank 10 failed, try rank 11
    if best_result is None:
        for restart in range(20):
            if time.time() > deadline:
                break
            res = try_rank(fallback_rank)
            if res:
                U, V, W = res
                rounded = round_and_verify(U, V, W)
                if rounded:
                    best_result = rounded
                    break
    
    # If still None, return schoolbook rank 12
    if best_result is None:
        # Schoolbook
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U_r, V_r, W_r = best_result
    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
