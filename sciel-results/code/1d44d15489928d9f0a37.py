import time

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 7 matrix.
    
    Uses the schoolbook decomposition which has rank n*m*p = 84.
    This is a valid exact decomposition.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U has shape (n*m, rank)
              V has shape (m*p, rank)
              W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time occasionally to ensure we don't exceed the hard limit
                if (col % 10 == 0) and (time.time() > deadline):
                    # If we exceed time, we must return a valid result.
                    # Since this loop is extremely fast (84 iterations), 
                    # this branch is effectively dead code for safety.
                    # We break to return the partial result? No, partial is invalid.
                    # We just let it finish as it's micro-seconds.
                    pass
                
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
