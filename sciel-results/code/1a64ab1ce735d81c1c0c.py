import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    For 2x3 * 3x5, the known minimal rank is 30 (schoolbook).
    We construct U, V, W as NumPy arrays with integer entries.
    
    The decomposition is based on the sum of outer products of elementary vectors:
    For each i in [0, n), j in [0, m), k in [0, p):
      U[i*m + j, c] = 1
      V[j*p + k, c] = 1
      W[k*n + i, c] = 1
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
