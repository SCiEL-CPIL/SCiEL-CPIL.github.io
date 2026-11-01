"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Schoolbook decomposition of matrix multiplication tensor for 2x5 * 5x5.
    Rank = 50.
    """
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    rank = n * m * p  # 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
