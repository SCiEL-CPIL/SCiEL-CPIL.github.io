def _find_decomposition(rank, max_iter=10000, tol=1e-8, seed=42):
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Precompute indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for it in range(max_iter):
        # Compute current error
        errors = []
        for (i, j, k) in indices:
            val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
            errors.append(val - 1.0)
        errors = np.array(errors)
        max_err = np.max(np.abs(errors))
        
        if max_err < tol:
            return U, V, W, max_err
        
        # Alternating least squares: fix two, solve for third
        # Solve for U: for each (i,j), we need sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
        # This is a linear system: U[i*m+j,:] * A_{i,j,k}^T = 1 for all k
        # where A_{i,j,k,r} = V[j*p+k,r] * W[k*n+i,r]
        
        # Actually, let's do a simpler update: use gradient descent
        lr = 0.01
        for (i, j, k) in indices:
            u_row = U[i*m+j]
            v_row = V[j*p+k]
            w_row = W[k*n+i]
            
            # Gradient with respect to U[i*m+j, :]
            grad_u = v_row * w_row * (np.dot(np.dot(u_row, v_row), w_row) - 1.0)
            U[i*m+j] -= lr * grad_u
            
            # Gradient with respect to V[j*p+k, :]
            grad_v = u_row * w_row * (np.dot(np.dot(u_row, v_row), w_row) - 1.0)
            V[j*p+k] -= lr * grad_v
            
            # Gradient with respect to W[k*n+i, :]
            grad_w = u_row * v_row * (np.dot(np.dot(u_row, v_row), w_row) - 1.0)
            W[k*n+i] -= lr * grad_w
    
    # Return final error
    errors = []
    for (i, j, k) in indices:
        val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
        errors.append(val - 1.0)
    errors = np.array(errors)
    return U, V, W, np.max(np.abs(errors))
