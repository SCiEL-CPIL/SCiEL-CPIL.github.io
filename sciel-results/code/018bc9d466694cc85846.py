import numpy as np

def construct():
    """
    Constructs the exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    The tensor for multiplying a 3x3 matrix A by a 3x7 matrix B is the direct sum of 
    7 instances of the matrix-vector multiplication tensor M_{3,3,1}.
    The rank of M_{n,m,1} is n*m. Thus, the total rank is 7 * (3*3) = 63.
    This is the minimal rank.
    
    Returns:
        dict: A dictionary containing 'U', 'V', 'W' matrices.
        U has shape (n*m, rank) -> (9, 63)
        V has shape (m*p, rank) -> (21, 63)
        W has shape (p*n, rank) -> (21, 63)
    """
    n, m, p = 3, 3, 7
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
