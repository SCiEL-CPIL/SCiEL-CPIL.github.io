import numpy as np

def construct():
    n, m, p = 2, 3, 8
    
    # The tensor for matrix multiplication under the trace convention 
    # tr(ABC) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} is the all-ones tensor.
    # The all-ones tensor has rank 1, given by U=1, V=1, W=1.
    
    rank = 1
    
    U = np.ones((n * m, rank), dtype=float)
    V = np.ones((m * p, rank), dtype=float)
    W = np.ones((p * n, rank), dtype=float)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
