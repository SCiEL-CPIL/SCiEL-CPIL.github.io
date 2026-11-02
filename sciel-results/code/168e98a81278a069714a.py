"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication by exploiting
    the row-wise structure (Kronecker-like separation of the n-dimension).
    For n=2, m=5, p=5, this decomposes into two independent 1x5x5 multiplications.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        # Each row i of A (size 1 x m) multiplies with B (m x p) to produce row i of C (1 x p)
        # This is a 1 x m x p multiplication, which has rank m * p
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (Trace convention: W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
