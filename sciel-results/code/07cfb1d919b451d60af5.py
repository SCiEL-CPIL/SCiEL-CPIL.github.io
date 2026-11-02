"""Matrix multiplication 2x2x2: rank-7 Strassen decomposition."""
import numpy as np

def construct():
    """
    Returns a rank-7 tensor decomposition for 2x2x2 matrix multiplication.
    """
    rank = 7
    
    # Strassen's 7 products
    # U rows: a, b, c, d
    # V rows: e, f, g, h
    # W rows: C00, C10, C01, C11 (indexed as k*2+i)
    
    U = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 1, 0]
    ], dtype=float)
    
    V = np.array([
        [1, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 1, -1, -1, 0, -1, 1]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 0],
        [1, 1, 0, 0, 0, 0, -1],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, -1, 0, 1, 0]
    ], dtype=float)
    
    return {"U": U, "V": V, "W": W}
