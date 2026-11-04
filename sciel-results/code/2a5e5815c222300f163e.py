def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute the tensor: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention is:
    # (AB)[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # This is not a standard tensor contraction T[i,j,k].
    # It is: C[i,k] = sum_c W[k,i,c] * (U_c . A_i) * (V_c . B_k)
    # Where U_c is a vector of length n*m? No, U is (n*m, rank).
    # U[i*m+j, c] is the coefficient for A[i,j] in the c-th term.
    # So for fixed c, we have a matrix U_c of size n x m?
    # Let U_c[i, j] = U[i*m+j, c].
    # Let V_c[j, k] = V[j*p+k, c].
    # Let W_c[k, i] = W[k*n+i, c].
    # Then C = sum_c W_c * (U_c * A) * (B * V_c^T)? No.
    # The term is scalar: (sum_j U[i,j,c] A[i,j]) is not right.
    # U[i*m+j, c] multiplies A[i,j].
    # So for fixed c, the "A-part" is a scalar? No.
    # The decomposition is:
    # (AB)[i,k] = sum_c ( sum_j U[i*m+j, c] A[i,j] ) * ( sum_l V[j*p+k, c] B[j,k] ) * W[k*n+i, c]
    # This formula is wrong in the code comments.
    # Correct formula:
    # (AB)[i,k] = sum_c W[k*n+i, c] * ( sum_j U[i*m+j, c] A[i,j] ) * ( sum_l V[l*p+k, c] B[l,k] )
    # Wait, the sums over j and l are independent?
    # If so, it factors into (sum_j ...) * (sum_l ...).
    # This implies the c-th term is a product of two linear forms in A and B, multiplied by W.
    # This is the standard definition.
    
    # Let's test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using decomposition
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            u_vals = np.zeros(rank)
            for j in range(m):
                u_vals += U[i*m+j, :] * A[i, j]
            v_vals = np.zeros(rank)
            for l in range(m):
                v_vals += V[l*p+k, :] * B[l, k]
            C_decomp[i, k] = np.dot(W[k*n+i, :], u_vals * v_vals)
    
    err = np.max(np.abs(C_decomp - C_expected))
    return err < tol, err
