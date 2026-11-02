"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal decomposition for 2x3 * 3x5 matrix multiplication.
    
    The matrix multiplication tensor for (n x m) * (m x p) has:
    - U: shape (n*m, rank)
    - V: shape (m*p, rank)  
    - W: shape (p*n, rank)
    
    with the trace convention W[k*n + i, r].
    
    The tensor T[i, j, j', k] = delta_{j, j'} (for the multiplication C[i,k] = sum_j A[i,j]*B[j,k])
    
    We can achieve rank = m*p = 15 using the following construction:
    - Let c = (j', k) where j' in [0, m) and k in [0, p)
    - U[i*m + j, (j', k)] = delta_{j, j'}
    - V[j'*m + k, (j'', k'')] = delta_{j', j''} * delta_{k, k''} (i.e., identity in the (j',k) space)
    - W[k*n + i, (j', k'')] = 1
    
    This gives sum over (j', k'') of U[i*m+j, (j',k'')] * V[j'*m+k, (j',k'')] * W[k*n+i, (j',k'')]
    = sum_{j',k''} delta_{j,j'} * delta_{j',j'} * delta_{k,k''} * 1
    = sum_{j',k''} delta_{j,j'} * delta_{j',j'} * delta_{k,k''}
    = delta_{j,j'} (where j'=j) * 1 (since k''=k)
    = delta_{j,j}... 
    
    Wait, let me re-derive. The product is:
    sum_c U[i*m+j, c] * V[j'_m + k, c] * W[k*n + i, c]
    
    where c indexes (j'', k'').
    
    U[i*m+j, (j'', k'')] = delta_{j, j''}
    V[j'_m + k, (j'', k'')] = delta_{j', j''} * delta_{k, k''}
    W[k*n + i, (j'', k'')] = 1
    
    So the sum is:
    sum_{j'', k''} delta_{j, j''} * delta_{j', j''} * delta_{k, k''} * 1
    = delta_{j, j'} * 1  (since j'' must equal both j and j', and k'' must equal k)
    = delta_{j, j'}
    
    This is exactly what we need for matrix multiplication!
    """
    n, m, p = DIMENSIONS
    rank = m * p  # 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j_prime in range(m):
        for k in range(p):
            col = j_prime * p + k  # column index in U, V, W
            
            # U[i*m + j, col] = delta_{j, j_prime}
            for i in range(n):
                for j in range(m):
                    if j == j_prime:
                        U[i * m + j][col] = 1
            
            # V[j_prime * m + k, col] = delta_{j_prime, j_prime} * delta_{k, k} = 1
            # But wait, V's rows are indexed by (j', k) where j' is the row of B
            # V[row, col] where row = j' * p + k (B has m*p entries indexed as j*p+k)
            # We need V[j' * p + k, (j'', k'')] = delta_{j', j''} * delta_{k, k''}
            # So V is just the identity matrix in this indexing!
            V[j_prime * p + k][col] = 1
            
            # W[k*n + i, col] = 1
            for i in range(n):
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

