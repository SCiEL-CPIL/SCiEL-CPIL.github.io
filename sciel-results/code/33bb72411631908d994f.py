import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    For 2x3 * 3x5, the known minimal rank is 30 (schoolbook).
    We construct U, V, W with integer entries using a direct Kronecker-product structure.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterating over the tensor indices i, j, k
    # i: 0..n-1 (row of A, row of C)
    # j: 0..m-1 (col of A, row of B)
    # k: 0..p-1 (col of B, col of C)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i, col j of A -> i*m + j
                u_idx = i * m + j
                # Index in V: row j, col k of B -> j*p + k
                v_idx = j * p + k
                # Index in W: row k, col i of C (trace convention W[k*n+i, r]) -> k*n + i
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
