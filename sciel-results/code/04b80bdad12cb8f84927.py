import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns the schoolbook decomposition with rank 72.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Rank is n * m * p
    R = n * m * p
    
    # Initialize matrices
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                idx_U = i * m + j
                # B index: j*p + k
                idx_V = j * p + k
                # C index: k*n + i (as per task convention)
                idx_W = k * n + i
                
                U[idx_U, col] = 1.0
                V[idx_V, col] = 1.0
                W[idx_W, col] = 1.0
                
                col += 1
    
    # Convert to lists of lists as required by the interface example (or return np arrays if accepted? 
    # The example returns lists of lists: {'U': U, 'V': V, 'W': W} where U is a list)
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The seed program returns lists of lists. I should match that to be safe.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
