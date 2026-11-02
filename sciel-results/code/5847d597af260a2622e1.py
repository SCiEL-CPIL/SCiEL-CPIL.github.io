"""
Optimized program for 2x2 matrix multiplication tensor decomposition.
Returns the rank-7 Strassen decomposition directly.
"""
import numpy as np

DIMENSIONS = [2, 2, 2]

def construct():
    """
    Returns the rank-minimal (rank 7) exact tensor decomposition for 2x2 matrix multiplication.
    """
    # Strassen's rank-7 decomposition
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    # Coefficients for A (U)
    # Row indices: A[0,0], A[0,1], A[1,0], A[1,1]
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # Coefficients for B (V)
    # Row indices: B[0,0], B[0,1], B[1,0], B[1,1]
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    # Coefficients for C (W)
    # Row indices: k*n + i. n=2.
    # k=0, i=0 -> 0 (C[0,0])
    # k=0, i=1 -> 1 (C[1,0])
    # k=1, i=0 -> 2 (C[0,1])
    # k=1, i=1 -> 3 (C[1,1])
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
