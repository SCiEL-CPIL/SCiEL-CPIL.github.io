"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If we broke early, we might have an incomplete decomposition.
    # However, for n=3, m=3, p=6, the loop is tiny (54 iterations).
    # The deadline check is just to satisfy the directive.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
