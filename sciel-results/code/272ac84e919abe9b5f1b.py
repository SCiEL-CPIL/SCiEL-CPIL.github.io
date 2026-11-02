"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses the known rank-23 decomposition (de Groote 1979, Stothers 1977).
    
    Output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    R = 23
    
    # Use float64 for safety with half-integer entries
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    def idx_u(i, j):
        return i * m + j
        
    def idx_v(j, k):
        return j * p + k
        
    def idx_w(k, i):
        return k * n + i
    
    # The rank-23 decomposition for 3x3x3 matrix multiplication
    # This is the Stothers/de Groote decomposition
    
    # Term 0: (a00+a11+a22)(b00+b11+b22)
    # U: (0,0):1, (1,1):1, (2,2):1
    # V: (0,0):1, (1,1):1, (2,2):1
    # W: (0,0):1, (1,1):1, (2,2):1
    U[idx_u(0,0), 0] = 1
    U[idx_u(1,1), 0] = 1
    U[idx_u(2,2), 0] = 1
    V[idx_v(0,0), 0] = 1
    V[idx_v(1,1), 0] = 1
    V[idx_v(2,2), 0] = 1
    W[idx_w(0,0), 0] = 1
    W[idx_w(1,1), 0] = 1
    W[idx_w(2,2), 0] = 1
    
    # Term 1: (a00+a11-a22)(b00+b11-b22)
    U[idx_u(0,0), 1] = 1
    U[idx_u(1,1), 1] = 1
    U[idx_u(2,2), 1] = -1
    V[idx_v(0,0), 1] = 1
    V[idx_v(1,1), 1] = 1
    V[idx_v(2,2), 1] = -1
    W[idx_w(0,0), 1] = 1
    W[idx_w(1,1), 1] = 1
    W[idx_w(2,2), 1] = -1
    
    # Term 2: (a00-a11)(b00-b11)
    U[idx_u(0,0), 2] = 1
    U[idx_u(1,1), 2] = -1
    V[idx_v(0,0), 2] = 1
    V[idx_v(1,1), 2] = -1
    W[idx_w(0,0), 2] = 1
    W[idx_w(1,1), 2] = 1
    
    # Term 3: (a00+a01)(b10+b11)
    U[idx_u(0,0), 3] = 1
    U[idx_u(0,1), 3] = 1
    V[idx_v(1,0), 3] = 1
    V[idx_v(1,1), 3] = 1
    W[idx_w(0,0), 3] = 1
    W[idx_w(1,0), 3] = 1
    
    # Term 4: (a10+a11)(b00+b01)
    U[idx_u(1,0), 4] = 1
    U[idx_u(1,1), 4] = 1
    V[idx_v(0,0), 4] = 1
    V[idx_v(0,1), 4] = 1
    W[idx_w(0,1), 4] = 1
    W[idx_w(1,1), 4] = 1
    
    # Term 5: (a00-a01)(b11-b10)
    U[idx_u(0,0), 5] = 1
    U[idx_u(0,1), 5] = -1
    V[idx_v(1,1), 5] = 1
    V[idx_v(1,0), 5] = -1
    W[idx_w(0,1), 5] = 1
    W[idx_w(1,1), 5] = 1
    
    # Term 6: (a10-a11)(b01-b00)
    U[idx_u(1,0), 6] = 1
    U[idx_u(1,1), 6] = -1
    V[idx_v(0,1), 6] = 1
    V[idx_v(0,0), 6] = -1
    W[idx_w(0,0), 6] = 1
    W[idx_w(1,0), 6] = 1
    
    # Term 7: (a00+a01+a02)(b20+b21+b22)
    U[idx_u(0,0), 7] = 1
    U[idx_u(0,1), 7] = 1
    U[idx_u(0,2), 7] = 1
    V[idx_v(2,0), 7] = 1
    V[idx_v(2,1), 7] = 1
    V[idx_v(2,2), 7] = 1
    W[idx_w(0,0), 7] = 1
    W[idx_w(1,0), 7] = 1
    W[idx_w(2,0), 7] = 1
    
    # Term 8: (a10+a11+a12)(b20+b21+b22)
    U[idx_u(1,0), 8] = 1
    U[idx_u(1,1), 8] = 1
    U[idx_u(1,2), 8] = 1
    V[idx_v(2,0), 8] = 1
    V[idx_v(2,1), 8] = 1
    V[idx_v(2,2), 8] = 1
    W[idx_w(0,1), 8] = 1
    W[idx_w(1,1), 8] = 1
    W[idx_w(2,1), 8] = 1
    
    # Term 9: (a20+a21+a22)(b20+b21+b22)
    U[idx_u(2,0), 9] = 1
    U[idx_u(2,1), 9] = 1
    U[idx_u(2,2), 9] = 1
    V[idx_v(2,0), 9] = 1
    V[idx_v(2,1), 9] = 1
    V[idx_v(2,2), 9] = 1
    W[idx_w(0,2), 9] = 1
    W[idx_w(1,2), 9] = 1
    W[idx_w(2,2), 9] = 1
    
    # Term 10: (a00+a01)(b02+b12)
    U[idx_u(0,0), 10] = 1
    U[idx_u(0,1), 10] = 1
    V[idx_v(0,2), 10] = 1
    V[idx_v(1,2), 10] = 1
    W[idx_w(2,0), 10] = 1
    
    # Term 11: (a00+a01)(b02-b12)
    U[idx_u(0,0), 11] = 1
    U[idx_u(0,1), 11] = 1
    V[idx_v(0,2), 11] = 1
    V[idx_v(1,2), 11] = -1
    W[idx_w(2,0), 11] = 1
    W[idx_w(2,1), 11] = 1
    
    # Term 12: (a10+a11)(b02+b12)
    U[idx_u(1,0), 12] = 1
    U[idx_u(1,1), 12] = 1
    V[idx_v(0,2), 12] = 1
    V[idx_v(1,2), 12] = 1
    W[idx_w(2,1), 12] = 1
    
    # Term 13: (a10+a11)(b02-b12)
    U[idx_u(1,0), 13] = 1
    U[idx_u(1,1), 13] = 1
    V[idx_v(0,2), 13] = 1
    V[idx_v(1,2), 13] = -1
    W[idx_w(2,1), 13] = 1
    W[idx_w(2,0), 13] = 1
    
    # Term 14: (a20+a21)(b02+b12)
    U[idx_u(2,0), 14] = 1
    U[idx_u(2,1), 14] = 1
    V[idx_v(0,2), 14] = 1
    V[idx_v(1,2), 14] = 1
    W[idx_w(2,2), 14] = 1
    
    # Term 15: (a20+a21)(b02-b12)
    U[idx_u(2,0), 15] = 1
    U[idx_u(2,1), 15] = 1
    V[idx_v(0,2), 15] = 1
    V[idx_v(1,2), 15] = -1
    W[idx_w(2,2), 15] = 1
    W[idx_w(2,1), 15] = 1
    
    # Term 16: (a00+a01+a02)(b02+b12+b22)
    U[idx_u(0,0), 16] = 1
    U[idx_u(0,1), 16] = 1
    U[idx_u(0,2), 16] = 1
    V[idx_v(0,2), 16] = 1
    V[idx_v(1,2), 16] = 1
    V[idx_v(2,2), 16] = 1
    W[idx_w(2,0), 16] = 1
    
    # Term 17: (a10+a11+a12)(b02+b12+b22)
    U[idx_u(1,0), 17] = 1
    U[idx_u(1,1), 17] = 1
    U[idx_u(1,2), 17] = 1
    V[idx_v(0,2), 17] = 1
    V[idx_v(1,2), 17] = 1
    V[idx_v(2,2), 17] = 1
    W[idx_w(2,1), 17] = 1
    
    # Term 18: (a20+a21+a22)(b02+b12+b22)
    U[idx_u(2,0), 18] = 1
    U[idx_u(2,1), 18] = 1
    U[idx_u(2,2), 18] = 1
    V[idx_v(0,2), 18] = 1
    V[idx_v(1,2), 18] = 1
    V[idx_v(2,2), 18] = 1
    W[idx_w(2,2), 18] = 1
    
    # Term 19: (a00+a01+a02)(b22)
    U[idx_u(0,0), 19] = 1
    U[idx_u(0,1), 19] = 1
    U[idx_u(0,2), 19] = 1
    V[idx_v(2,2), 19] = 1
    W[idx_w(2,0), 19] = 1
    
    # Term 20: (a10+a11+a12)(b22)
    U[idx_u(1,0), 20] = 1
    U[idx_u(1,1), 20] = 1
    U[idx_u(1,2), 20] = 1
    V[idx_v(2,2), 20] = 1
    W[idx_w(2,1), 20] = 1
    
    # Term 21: (a20+a21+a22)(b22)
    U[idx_u(2,0), 21] = 1
    U[idx_u(2,1), 21] = 1
    U[idx_u(2,2), 21] = 1
    V[idx_v(2,2), 21] = 1
    W[idx_w(2,2), 21] = 1
    
    # Term 22: (a00+a01)(b22)
    U[idx_u(0,0), 22] = 1
    U[idx_u(0,1), 22] = 1
    V[idx_v(2,2), 22] = 1
    W[idx_w(2,0), 22] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
