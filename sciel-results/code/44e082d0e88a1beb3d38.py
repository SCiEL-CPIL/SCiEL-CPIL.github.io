import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 1
    
    U = np.ones((n * m, rank), dtype=int)
    V = np.ones((m * p, rank), dtype=int)
    W = np.ones((p * n, rank), dtype=int)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
