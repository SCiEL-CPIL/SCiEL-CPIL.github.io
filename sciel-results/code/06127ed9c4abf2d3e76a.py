# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Check if we have time to do anything fancy (we don't, but good practice)
    if time.time() > deadline:
        pass # Fallback to schoolbook immediately if somehow slow (impossible here)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U indexes (i,j) from input A
                U[i * m + j][col] = 1
                # V indexes (j,k) from input B
                V[j * p + k][col] = 1
                # W indexes (k,i) from output C, following W[k*n+i, r] convention
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
