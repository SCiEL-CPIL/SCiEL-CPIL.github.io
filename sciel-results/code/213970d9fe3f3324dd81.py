"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication C = AB has entries T[i,j,k] = 1
    # for all i in 1..m, j in 1..n, k in 1..p.
    # The rank of the all-ones tensor of size (m*n) x (n*p) x (p*m) is 
    # min(m*n, n*p, p*m).
    # We use the uniform decomposition.
    rank = min(n * m, m * p, p * n)
    
    U = [[1.0] * rank for _ in range(n * m)]
    V = [[1.0] * rank for _ in range(m * p)]
    W = [[1.0 / rank] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
