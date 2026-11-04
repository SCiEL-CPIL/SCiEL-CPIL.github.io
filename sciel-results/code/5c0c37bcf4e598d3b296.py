"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-27 exact tensor decomposition for 3x3 matrix multiplication.
    
    This uses the standard schoolbook algorithm decomposition.
    The decomposition is exact with integer coefficients.
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    import time
    deadline = time.time() + 120
    
    n, m, p = DIMENSIONS  # 3, 3, 3
    rank = 27
    
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]  # 9 x 27
    V = [[0] * rank for _ in range(m * p)]  # 9 x 27
    W = [[0] * rank for _ in range(p * n)]  # 9 x 27
    
    # Schoolbook decomposition:
    # Each term c corresponds to a unique (i, j, k) triple.
    # c = i * (m * p) + j * p + k
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
