"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication.
    Rank = n * m * p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    # V: (m*p, rank)
    V = [[0.0] * rank for _ in range(m * p)]
    # W: (p*n, rank)
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for U: row i*m + j
                u_idx = i * m + j
                # Index for V: row j*p + k
                v_idx = j * p + k
                # Index for W: row k*n + i
                w_idx = k * n + i
                
                U[u_idx][column] = 1.0
                V[v_idx][column] = 1.0
                W[w_idx][column] = 1.0
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
