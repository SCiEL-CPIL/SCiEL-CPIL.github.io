"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook (rank-27) decomposition for 3x3 matrix multiplication.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 for all (i,j,k) in {0,1,2}^3.
    A rank-1 term is u ⊗ v ⊗ w where:
      - u is supported on A-indices (i*m + j)
      - v is supported on B-indices (j*p + k)  
      - w is supported on C-indices (k*n + i) per the trace convention W[k*n+i]
    
    For each (i,j,k), the term u_{i,j} * v_{j,k} * w_{k,i} must equal 1.
    The schoolbook uses one rank-1 term per (i,j,k) triple, giving rank 27.
    """
    n, m, p = 3, 3, 3
    rank = n * m * p  # 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A-index: i*m + j
                # B-index: j*p + k
                # C-index: k*n + i (trace convention W[k*n+i])
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
