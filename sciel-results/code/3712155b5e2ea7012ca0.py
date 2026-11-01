def verify_rank_decomp(n, m, p, U, V, W):
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Check if sum_k U[:,k] V[:,k] W[:,k] == Tensor
    # Tensor T[i,j,k] = 1 if i,j,k valid else 0
    # Actually, we check if the product of matrices matches the multiplication tensor.
    # T = sum_k U[:,k] V[:,k]^T W[:,k]^T (tensor form)
    # Or simply: for random A, B, check if A @ B == sum_k (U[:,k]^T A) (V[:,k]^T B) W[:,k]^T
    
    import numpy as np
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_calc = np.zeros((n, p))
    rank = U.shape[1]
    for k in range(rank):
        u = U[:, k]
        v = V[:, k]
        w = W[:, k]
        
        # u corresponds to A flattening. u is length n*m.
        # u^T A means sum_{i,j} u[i*m+j] A[i,j]
        # This is a scalar.
        a_val = 0
        for i in range(n):
            for j in range(m):
                a_val += u[i*m+j] * A[i,j]
                
        b_val = 0
        for j in range(m):
            for l in range(p):
                b_val += v[j*p+l] * B[j,l]
                
        # w corresponds to C flattening. w is length p*n.
        # w[k*n+i] adds to C[i,k]
        # We add a_val * b_val * w to C
        for l in range(p):
            for i in range(n):
                C_calc[i,l] += a_val * b_val * w[l*n+i]
                
    if not np.allclose(C_expected, C_calc):
        return False
    return True
