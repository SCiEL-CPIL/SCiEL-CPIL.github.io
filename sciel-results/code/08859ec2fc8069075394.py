"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill matrices based on the decomposition:
    # r corresponds to (i, j, k)
    # U[i*m + j, r] = 1 if r encodes i, j
    # V[j*p + k, r] = 1 if r encodes j, k
    # W[k*n + i, r] = 1 if r encodes k, i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                
                # U index: i*m + j
                U[i * m + j][r] = 1
                
                # V index: j*p + k
                V[j * p + k][r] = 1
                
                # W index: k*n + i
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
