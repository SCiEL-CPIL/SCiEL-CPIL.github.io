def _try_svd_search(n, m, p, deadline, max_rank=70, min_rank=65):
    # Construct the full MM tensor
    # T has shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Unfold to matrices
    # U shape (n*m, R)
    # V shape (m*p, R)
    # W shape (p*n, R)
    
    for R in range(max_rank, min_rank - 1, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize U, V, W from SVD of unfolded tensors
        # T_U shape (n*m, m*p * p*n)
        T_U = T.reshape(n*m, -1)
        # SVD is expensive, use truncated SVD or random projection
        # For small R, we can use random initialization and ALS
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # ALS refinement
            for _ in range(20):
                # Update U
                # T_U = U @ (V * W)^T ? No.
                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # Fix V, W. Solve for U.
                # T_U (n*m, m*p*p*n)
                # T_U[i, j*p*n + k] = sum_r U[i,r] V[j,r] W[k,r]
                # This is a linear system for U[i, :]
                # Let M_U[i, r] = sum_{j,k} T[i,j,k] V[j,r] W[k,r] ? No.
                # We want to minimize || T - U V^T W ... ||
                
                # Efficient ALS for 3-way tensor:
                # 1. Compute M_U = T_U @ (V * W) ???
                # T_U (n*m, m*p * p*n)
                # We can reshape T to (n*m, m*p, p*n)
                # Contract with V and W?
                
                # Update U:
                # U_new = argmin_U || T - sum_r U[:,r] V[:,r]^T W[:,r]^T ||_F
                # This is equivalent to:
                # For each i, U[i,:] = (T_i * (V*W)) / (V*W . V*W) ? No.
                
                # Let's use the standard ALS update:
                # M_U = T_U @ (V * W) is not quite right due to dimensions.
                
                # T_U (n*m, m*p, p*n)
                # M_U[i, r] = sum_{j,k} T[i,j,k] V[j,r] W[k,r]
                # We can compute this as:
                # M_U = np.einsum('ijk,jr,kr->ir', T, V, W)
                # Then U = M_U / (V*W . V*W) ? No.
                # The denominator is sum_{j,k} V[j,r]^2 W[k,r]^2 ? No.
                # The term is U[i,r] V[j,r] W[k,r].
                # Gradient wrt U[i,r] is -2 sum_{j,k} T[i,j,k] V[j,r] W[k,r] + 2 U[i,r] sum_{j,k} V[j,r]^2 W[k,r]^2
                # So U[i,r] = (sum_{j,k} T[i,j,k] V[j,r] W[k,r]) / (sum_{j,k} V[j,r]^2 W[k,r]^2)
                
                M_U = np.einsum('ijk,jr,kr->ir', T, V, W, optimize=True)
                denom_U = np.einsum('jr,kr->r', V*V, W*W)
                U = M_U / (denom_U + 1e-10)
                
                # Update V
                M_V = np.einsum('ijk,ir,kr->jr', T, U, W, optimize=True)
                denom_V = np.einsum('ir,kr->r', U*U, W*W)
                V = M_V / (denom_V + 1e-10)
                
                # Update W
                M_W = np.einsum('ijk,ir,jr->kr', T, U, V, optimize=True)
                denom_W = np.einsum('ir,jr->r', U*U, V*V)
                W = M_W / (denom_W + 1e-10)
                
            # Round to half-integers
            U_round = np.round(U * 2.0) / 2.0
            V_round = np.round(V * 2.0) / 2.0
            W_round = np.round(W * 2.0) / 2.0
            
            # Verify on full tensor
            T_rec = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round, optimize=True)
            if np.allclose(T_rec, T, atol=1e-9):
                return U_round, V_round, W_round, R
                
    return None
