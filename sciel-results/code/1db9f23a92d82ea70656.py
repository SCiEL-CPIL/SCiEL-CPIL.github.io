import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Tensor: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Precompute reshaped matrices for least squares
    # For W solve: sum_d U[a,d]*V[b,d] * W[c,d] = T[a,b,c]
    # Reshape: A_w[a*b + a', c] = sum_d U[a,d]*V[b,d] where a' = a*m*p + b... 
    # Actually: T_w[a*(m*p) + b, c] = T[a,b,c], shape (n*m*m*p, p*n)
    T_w = T.reshape(n*m * m*p, p*n)
    
    # For V solve: sum_d U[a,d]*W[c,d] * V[b,d] = T[a,b,c]
    # T_v[a*(p*n) + c, b] = T[a,b,c], shape (n*m*p*n, m*p)
    T_perm_v = np.transpose(T, (0, 2, 1))
    T_v = T_perm_v.reshape(n*m * p*n, m*p)
    
    # For U solve: sum_d V[b,d]*W[c,d] * U[a,d] = T[a,b,c]
    # T_u[b*(p*n) + c, a] = T[a,b,c], shape (m*p*p*n, n*m)
    T_perm_u = np.transpose(T, (1, 2, 0))
    T_u = T_perm_u.reshape(m*p * p*n, n*m)
    
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    def try_rank(r):
        np.random.seed(42)
        
        for attempt in range(20):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(200):
                if time.time() > deadline:
                    return None
                
                # Solve for W: for each c, sum_d (sum_{a,b} U[a,d]*V[b,d]*T[a,b,c]) ... 
                # Actually: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
                # Fix U,V: for each c, W[c,:] = argmin ||T[:, :, c] - U*V^T * W[c,:]||
                # Reshape: T_w[ab, c] = T[a,b,c]
                # A_w[ab, d] = U[a,d]*V[b,d] where ab = a*(m*p)+b
                A_w = (U[:, :, None] * V[None, :, :]).reshape(n*m * m*p, r)
                W_sol, _, _, _ = np.linalg.lstsq(A_w, T_w, rcond=None)
                W = W_sol.T  # (p*n, r)
                
                # Solve for V: T[a,b,c] = sum_d U[a,d]*W[c,d]*V[b,d]
                # Fix U,W: for each b, V[b,:] = argmin ||T[:, b, :] - U*W^T * V[b,:]||
                # T_v[ac, b] = T[a,c,b] (transposed)
                A_v = (U[:, :, None] * W[None, :, :]).reshape(n*m * p*n, r)
                V_sol, _, _, _ = np.linalg.lstsq(A_v, T_v, rcond=None)
                V = V_sol.T  # (m*p, r)
                
                # Solve for U: T[a,b,c] = sum_d V[b,d]*W[c,d]*U[a,d]
                # Fix V,W: for each a, U[a,:] = argmin ||T[a, :, :] - V*W^T * U[a,:]||
                # T_u[bc, a] = T[b,c,a] (transposed)
                A_u = (V[:, :, None] * W[None, :, :]).reshape(m*p * p*n, r)
                U_sol, _, _, _ = np.linalg.lstsq(A_u, T_u, rcond=None)
                U = U_sol.T  # (n*m, r)
                
                # Check convergence
                T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
                error = np.linalg.norm(T - T_recon)
                
                if error < 1e-10:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    T_recon_r = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
                    error_r = np.linalg.norm(T - T_recon_r)
                    
                    if error_r < 1e-10:
                        # Verify exactly
                        if np.allclose(T_recon_r, T, atol=1e-10):
                            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
                    
                    # Try integer rounding
                    U_r2 = np.round(U)
                    V_r2 = np.round(V)
                    W_r2 = np.round(W)
                    T_recon_r2 = np.einsum('ad,bd,cd->abc', U_r2, V_r2, W_r2)
                    if np.allclose(T_recon_r2, T, atol=1e-10):
                        return (U_r2.tolist(), V_r2.tolist(), W_r2.tolist())
            
            # After all iterations, check if we got close enough
            T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
            error = np.linalg.norm(T - T_recon)
            if error < 1e-8:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_recon_r = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
                if np.allclose(T_recon_r, T, atol=1e-8):
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        return None
    
    result = try_rank(14)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 16
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
