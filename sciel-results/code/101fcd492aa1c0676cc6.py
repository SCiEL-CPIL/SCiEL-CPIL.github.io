"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x4 matrix.
    
    The minimal rank for M_{3,4,4} is 48, achieved by the schoolbook algorithm.
    The decomposition is:
    U: shape (12, 48)
    V: shape (16, 48)
    W: shape (12, 48)
    
    Indexing conventions:
    U[i*m + j, t]
    V[j*p + k, t]
    W[k*n + i, t]
    
    where i in [0, n), j in [0, m), k in [0, p), and t in [0, rank).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
