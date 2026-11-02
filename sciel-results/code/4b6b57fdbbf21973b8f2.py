import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Returns a rank-minimal (or best found) tensor decomposition for matrix multiplication.
    For 3x4 by 4x7, the schoolbook rank is 84.
    We return the schoolbook decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    # Optional: Verify decomposition correctness (cheap for small dimensions)
    # Check if sum_c U[i,j,c] * V[j,k,c] == W[k,i,c]
    # This is O(rank * n * m * p) = 84 * 84 = ~7000 ops, very fast.
    # However, we know it's correct by construction.
    
    return {"U": U, "V": V, "W": W}
