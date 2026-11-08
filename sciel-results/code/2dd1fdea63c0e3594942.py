"""Direct rank-216 identity (schoolbook) decomposition for 6x6x6 matrix multiplication."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: ensure we finish well within the 120s evaluation window.
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 216
    
    # Build U, V, W as lists of lists with integer entries (0 or 1)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i*m + j
                U[i * m + j][column] = 1
                # V index: row j*p + k
                V[j * p + k][column] = 1
                # W index: row k*n + i (trace convention W[k*n + i, r])
                W[k * n + i][column] = 1
                column += 1
        
        # Check deadline (unlikely to be hit for this small problem)
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
