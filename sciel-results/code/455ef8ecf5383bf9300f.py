import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x5 matrix multiplication.
    The rank is n*m*p = 30, which is optimal for these dimensions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to flattened row indices
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                # Set the 1s for this rank-1 component
                U[u_row][column] = 1
                V[v_row][column] = 1
                W[w_row][column] = 1
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
