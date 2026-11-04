"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition for 4x4x5
    # Rank = 4 * 4 * 5 = 80
    rank = 80
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] * B[j,k] contributes to C[i,k]
                u_flat = i * m + j
                v_flat = j * p + k
                w_flat = k * n + i
                
                U[u_flat][col] = 1
                V[v_flat][col] = 1
                W[w_flat][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
