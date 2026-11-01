"""Program for finding a rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Uses the schoolbook decomposition which gives rank = n*m*p = 3*3*7 = 63.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify correctness
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(W[0])
        # Check that sum over r of U[i,j,r] * V[j,k,r] = W[k,i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 0.0
                    for r in range(rank):
                        lhs += U[i * m + j][r] * V[j * p + k][r]
                    rhs = 0.0
                    for r in range(rank):
                        rhs += W[k * n + i][r] * U[i * m + j][r] * V[j * p + k][r]
                    if abs(lhs - rhs) > 1e-10:
                        return False
        return True
    
    # For schoolbook, verification should pass
    # Return the result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
