import numpy as np

def construct():
    n, m, p = 3, 5, 6
    
    # The structure tensor for matrix multiplication C = A @ B
    # is T[i, j, k] = 1 for all i, j, k.
    # This tensor can be decomposed with rank 1:
    # T[i, j, k] = U[i, 0] * V[j, 0] * W[k, 0]
    # where U, V, W are vectors of 1s.
    
    # However, the interface requires:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    rank = 1
    
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
