"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank of M_{n x m, m x p} is p * m if we decompose into p independent M_{n x m, m x 1} multiplications.
    # For M_{3 x 3, 3 x 1}, the rank is 3.
    # Total rank = 8 * 3 = 24.
    rank = p * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for j in range(m):
            c = k * m + j
            
            # V index: j * p + k
            V[j * p + k][c] = 1
            
            for i in range(n):
                # U index: i * m + j
                U[i * m + j][c] = 1
                # W index: k * n + i
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
