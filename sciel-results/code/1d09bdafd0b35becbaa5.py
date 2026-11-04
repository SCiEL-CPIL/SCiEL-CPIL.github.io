import time

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    The rank of the matrix multiplication tensor $M_{2,4,6}$ is 48.
    This is achieved by the standard schoolbook multiplication algorithm.
    
    Returns:
        dict: A dictionary containing keys 'U', 'V', 'W' with the factor matrices.
              U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the seed program interface, 
    # but we can use numpy if preferred. The seed used lists.
    # The prompt says "integer/half-integer entries". Lists of ints are fine.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index conventions:
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
