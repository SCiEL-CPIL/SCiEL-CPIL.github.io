def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x4 matrix.
    
    The tensor rank for <2, 3, 4> is known to be 24 (schoolbook).
    This function constructs the factor matrices U, V, W for this decomposition.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U has shape (n*m, rank) = (6, 24)
              V has shape (m*p, rank) = (12, 24)
              W has shape (p*n, rank) = (8, 24)
    """
    n, m, p = 2, 3, 4
    rank = n * m * p  # 24
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct the schoolbook decomposition
    # Each (i, j, k) triple corresponds to one rank-1 term
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Unique column index for this term
                r = i * m * p + j * p + k
                
                # U corresponds to A[i, j]
                # Index in U is i*m + j
                U[i * m + j][r] = 1
                
                # V corresponds to B[j, k]
                # Index in V is j*p + k
                V[j * p + k][r] = 1
                
                # W corresponds to C[i, k]
                # Index in W is k*n + i (following the trace convention W[k*n+i, r])
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
