"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Decomposition of 3x3x6 into two 3x3x3 blocks
    # Rank = 9 + 9 = 18
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # First block: k = 0, 1, 2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
                if time.time() > deadline:
                    # Return what we have so far (though it won't be valid if incomplete)
                    # But this loop is tiny, so it's fine.
                    pass

    # Second block: k = 3, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + (k + 3)][col] = 1
                W[(k + 3) * n + i][col] = 1
                col += 1
                
                if time.time() > deadline:
                    pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
