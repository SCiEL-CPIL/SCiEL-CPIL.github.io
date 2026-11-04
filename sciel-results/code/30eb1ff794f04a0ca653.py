"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 8 decomposition: m ranks for each of the n rows
    rank = n * m
    
    # U shape: (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V shape: (m*p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W shape: (p*n, rank)
    W = [[0] * rank for _ in range(p * n)]
    
    # For each row i in 0..n-1, we use m ranks
    for i in range(n):
        offset = i * m
        for r in range(m):
            col = offset + r
            # U[i*m + r, col] = 1
            U[i * m + r][col] = 1
            
            # V[j*p + k, col] = 1 if j == r
            for k in range(p):
                V[r * p + k][col] = 1
                
            # W[k*n + i, col] = 1
            for k in range(p):
                W[k * n + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
