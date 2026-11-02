import numpy as np

def construct():
    n, m, p = 2, 4, 6
    # T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Target ranks
    ranks_to_try = [32, 34, 36, 38, 40, 41, 42]
    
    def try_rank(R, max_iters=50, tol=1e-8):
        np.random.seed(123)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        for _ in range(max_iters):
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    a_idx = i * m + j
                    t = T[i, j, :] # size p
                    M = np.zeros((p, R))
                    for r in range(R):
                        v_vec = V[j * p : (j + 1) * p, r]
                        w_vec = np.array([W[k * n + i, r] for k in range(p)])
                        M[:, r] = v_vec * w_vec
                    U_new[a_idx, :], _ = np.linalg.lstsq(M, t, rcond=None)
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    b_idx = j * p + k
                    t = T[:, j, k] # size n
                    M = np.zeros((n, R))
                    for r in range(R):
                        u_vec = np.array([U[i * m + j, r] for i in range(n)])
                        w_val = W[k * n + i, r] # Wait, W depends on i.
                        # This update is not straightforward because W[k*n+i, r] depends on i.
                        # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # For fixed j, k, and varying i:
                        # T[i, j, k] = sum_r (U[i*m+j, r] * W[k*n+i, r]) * V[j*p+k, r]
                        # Let A_r = U[i*m+j, r] * W[k*n+i, r]. This depends on i.
                        # So M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        for i in range(n):
                            M[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    V_new[b_idx, :], _ = np.linalg.lstsq(M, t, rcond=None)
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    c_idx = k * n + i
                    t = T[i, :, k] # size m
                    M = np.zeros((m, R))
                    for r in range(R):
                        for j in range(m):
                            M[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    W_new[c_idx, :], _ = np.linalg.lstsq(M, t, rcond=None)
            W = W_new
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error += (T[i, j, k] - val) ** 2
            if error < tol:
                # Round to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U_int[i * m + j, r] * V_int[j * p + k, r] * W_int[k * n + i, r]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return U_int, V_int, W_int, R
        return None

    for R in ranks_to_try:
        res = try_rank(R)
        if res:
            U, V, W, rank = res
            # Convert to lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to Strassen 42
    return construct_strassen()

def construct_strassen():
    # ... (copy from original) ...
