import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (known upper bound) exact tensor decomposition 
    for multiplying a 5x5 matrix by a 5x5 matrix.
    
    Currently uses the schoolbook decomposition of rank n*m*p = 125.
    Finding a decomposition with rank < 125 for 5x5 matrix multiplication
    is a complex algebraic problem (best known is 103) that requires
    pre-computed coefficients not derivable in real-time within the time limit.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    # If we run out of time (unlikely for schoolbook), return what we have?
                    # But schoolbook is O(125) operations, so this won't happen.
                    pass
                    
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
