def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check tensor identity: for all i,j,k
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i,j,k are valid indices
    # Actually the tensor for matrix multiplication C = A*B is:
    # T[i,k,j] = sum_l A[i,l] * B[l,k]
    # In our indexing: A is n x m, B is m x p, C is n x p
    # T[i, l, k] where i in 0..n-1, l in 0..m-1, k in 0..p-1
    # The rank-1 decomposition gives: sum_r U[i,l,r] * V[l,k,r] * W[k,i,r]
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So W index is (k*n + i), which means W[k,i] in natural indexing.
    # U shape is (n*m, rank) - so U[i*m + l, r] for i in 0..n-1, l in 0..m-1
    # V shape is (m*p, rank) - so V[l*p + k, r] for l in 0..m-1, k in 0..p-1
    # W shape is (p*n, rank) - so W[k*n + i, r] for k in 0..p-1, i in 0..n-1
    
    # The tensor identity is:
    # For all i (0..n-1), l (0..m-1), k (0..p-1):
    # sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+l][r] * V[l*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False, (i, l, k, s)
    return True, None
