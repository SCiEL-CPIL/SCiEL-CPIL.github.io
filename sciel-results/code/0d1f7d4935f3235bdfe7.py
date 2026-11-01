import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or better) tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Currently uses schoolbook decomposition (rank 63) which is guaranteed valid.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard (not strictly needed for static construction, but good practice)
    # start_time = time.time()
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
