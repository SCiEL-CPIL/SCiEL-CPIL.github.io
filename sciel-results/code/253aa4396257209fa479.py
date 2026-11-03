"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Direct decomposition: rank = m * n = 4 * 2 = 8
    # For each (i, j) pair, we have a rank-1 term that captures A[i,j] * B[j,:] contribution to C[i,:]
    # U[i*m+j, r] = 1 if r == i*m+j
    # V[j*p+k, r] = 1 if r corresponds to j (i.e., r // n == j) -- but we need V to depend on j,k
    # W[k*n+i, r] = 1 if r corresponds to i (i.e., r % n == i) -- but we need W to depend on k,i
    
    # Actually, let's think more carefully.
    # The tensor T_{i,j,k} = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k] with coeff 1)
    # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # Let r index pairs (i, j) where i in [0,n), j in [0,m)
    # So r = i*m + j, total rank = n*m = 8
    
    # U[i,j,r] = 1 if r == i*m + j, else 0
    # V[j,k,r] = 1 if r corresponds to some i' and j' where j' == j, else 0
    #   But r = i'*m + j', so V[j,k,r] = 1 if j' == j, i.e., r % m == j
    # W[k,i,r] = 1 if r corresponds to some i' and j' where i' == i, else 0
    #   i' = r // m, so W[k,i,r] = 1 if r // m == i
    
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Only r = i*m + j contributes: U=1, V: r%m = j ✓, W: r//m = i ✓
    # So sum = 1. Correct!
    
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            # V[j, k, r] = 1 for all k
            for k in range(p):
                V[j * p + k][r] = 1
            # W[k, i, r] = 1 for all k
            for k in range(p):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
