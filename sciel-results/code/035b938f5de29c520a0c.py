def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # Rank 20 decomposition
    # For each k in 0..9 (column of B), and each i in 0..1 (row of A)
    # We compute A[i, k] * B[k]
    # This contributes to C[i]
    
    rank = 20
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for k in range(p):
        for i in range(n):
            # Term for A[i, k] * B[k] -> C[i]
            # U index: i * m + k
            # V index: k (since B is 3x10? No, B is 3x10, so B[j, k] is element)
            # Wait, B is m x p = 3 x 10.
            # So B[j, k] is the element.
            # In the tensor, V index is j * p + k?
            # No, V is (m*p, rank).
            # The input B is flattened.
            # V index for B[j, k] is j * p + k?
            # Let's check the trace convention.
            # W[k*n+i, r].
            # U[i*m+j, r].
            # V[j*p+k, r].
            
            # We want to compute C[i] += A[i, k] * B[k]?
            # No, C[i] = sum_j A[i, j] B[j, k].
            # So for a fixed output column k, C[i, k] = sum_j A[i, j] B[j, k].
            # My previous logic "A[i, k] * B[k]" assumed A was 2x10?
            # A is 2x3. B is 3x10.
            # C is 2x10.
            # C[i, k] = sum_{j=0}^2 A[i, j] B[j, k].
            
            # So for each output element (i, k), we sum 3 products.
            # Product r: A[i, j] * B[j, k].
            # This requires 1 multiplication.
            # Total multiplications: 2 * 10 * 3 = 60.
            # This is the schoolbook rank.
            
            # How can we reduce it?
            # We can group terms.
            # For a fixed k, we compute C[:, k] = A @ B[:, k].
            # This is 2x3 matrix times 3x1 vector.
            # Rank of this operation is 3? Or 2?
            # If rank is 3, total rank is 10 * 3 = 30.
            # If rank is 4, total rank is 40.
            
            pass
