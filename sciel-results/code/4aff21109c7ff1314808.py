"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for jp in range(m * p):
        j = jp // p
        k = jp % p
        
        # U: for all i, U[(i*m + j), jp] = 1 if j' == j
        for i in range(n):
            U[i * m + j][jp] = 1
        
        # V: V[(j*p + k), jp] = 1 if k' == k
        V[j * p + k][jp] = 1
        
        # W: for all i', W[(k*n + i'), jp] = 1
        for i_prime in range(n):
            W[k * n + i_prime][jp] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
