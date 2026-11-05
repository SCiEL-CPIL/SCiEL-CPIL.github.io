"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Schoolbook decomposition for 3x3 * 3x4 matrix multiplication.
    Rank = 3 * 3 * 4 = 36.
    This is a valid baseline.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # C[i,k] += A[i,j] * B[j,k]
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
