"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Rank 24 construction: one term per (i,k) pair
    rank = n * p  # 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(rank):
        i = t // p
        k = t % p
        for j in range(m):
            U[i * m + j][t] = 1
            V[j * p + k][t] = 1
        W[k * n + i][t] = 1
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen for this simple construction
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
