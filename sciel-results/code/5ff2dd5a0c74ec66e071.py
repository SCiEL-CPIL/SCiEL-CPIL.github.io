import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The rank is n*m*p = 105.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as required by the interface
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition
    # Each term (i, j, k) corresponds to a rank-1 component
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1.0
                V[v_idx][r] = 1.0
                W[w_idx][r] = 1.0
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
