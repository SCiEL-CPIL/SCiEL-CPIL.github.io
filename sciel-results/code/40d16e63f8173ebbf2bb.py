import numpy as np

def construct():
    n, m, p = 3, 4, 5
    rank = 1
    U = np.ones((n * m, rank), dtype=float)
    V = np.ones((m * p, rank), dtype=float)
    W = np.ones((p * n, rank), dtype=float)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
