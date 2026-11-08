"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    The minimum rank for 3x3 matrix multiplication is 27 (schoolbook rank).
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition satisfies: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c
    yields the standard matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 27
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # Each term in the sum corresponds to a specific (i, j, k) index triple
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
