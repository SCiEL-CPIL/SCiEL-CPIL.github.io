def _construct_rank_mp(n, m, p):
    """
    Construct a rank m*p decomposition.
    Rank indices correspond to (j, k) pairs.
    r = j * p + k
    """
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # V is identity: V[j*p+k, r] = 1
            V[j * p + k][r] = 1
            
            # U: row i*m+j, col r. Non-zero if j matches row's j.
            # Since r encodes j, and row encodes j, they always match for the specific r?
            # No, U[i*m+j, r] should be 1 for all i, for the r corresponding to this j.
            # Wait, in the sum for a fixed (i, j, k), we pick r = j*p+k.
            # We need U[i*m+j, j*p+k] = 1.
            # Does U[i*m+j, r'] need to be 0 for r' with different j'?
            # If U[i*m+j, r'] is non-zero for r' with j' != j,
            # then in the sum for (i, j, k), the term with r' would be:
            # U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r']
            # V[j*p+k, r'] is 0 if r' != j*p+k (since V is identity).
            # So U can be non-zero anywhere, as long as V is identity?
            # If V is identity, V[row, r] is 1 only if row==r.
            # So for a fixed target (i, j, k), the sum is over r.
            # Term r is non-zero only if V[j*p+k, r] != 0.
            # Since V is identity, this only happens if r == j*p+k.
            # So we only care about the term r = j*p+k.
            # For this term, we need U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1.
            # We can set U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1.
            
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
