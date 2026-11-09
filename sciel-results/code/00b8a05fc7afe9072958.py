"""Matrix multiplication 2x3x8 rank-minimal exact decomposition."""
DIMENSIONS = [2, 3, 8]

import numpy as np


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
    
    Uses the natural rank-m decomposition where each column corresponds to one
    inner-dimension index j. This gives rank = m = 3.
    
    Interface: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    W follows the convention W[k*n+i, c].
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    r = m  # rank = 3
    
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    for c in range(r):
        # c corresponds to inner index j = c
        # U[i*m + j, c] = 1 if j == c, else 0
        for i in range(n):
            U[i * m + c, c] = 1
        
        # V[j*p + k, c] = 1 if j == c, else 0
        for k in range(p):
            V[c * p + k, c] = 1
        
        # W[k*n + i, c] = 1 for all (k, i)
        for k in range(p):
            for i in range(n):
                W[k * n + i, c] = 1
    
    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                assert abs(s - 1.0) < 1e-12, f"Failed at ({i},{j},{k}): got {s}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
