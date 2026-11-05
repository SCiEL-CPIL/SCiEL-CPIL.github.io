"""Matrix multiplication 3x5x7 - Kronecker product approach."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    
    # The tensor T_{n,m,p} can be seen as T_{n,1,1} (Kronecker) T_{1,m,p}
    # T_{n,1,1} has rank n (identity-like)
    # T_{1,m,p} has rank m*p (identity-like)
    # Total rank is n * m * p
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Constructing the Kronecker product structure
    # Index mapping for Kronecker product of (n,1,1) and (1,m,p)
    # Total indices:
    # i in [0, n)
    # j in [0, m)
    # k in [0, p)
    
    # The rank-1 terms are indexed by (i, j, k)
    # U[i, j] corresponds to coefficient of e_i in A, e_j in B?
    # Standard: U is (n*m) x rank. V is (m*p) x rank. W is (p*n) x rank.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This is the schoolbook term
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
