"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Returns a valid rank-26 decomposition using Strassen + border terms.
    The score is 27/rank, so rank 26 gives score ~1.0385.
    """
    n, m, p = DIMENSIONS
    R = 26
    
    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)
    
    def _iu(i, j):
        return i * m + j
        
    def _iv(j, k):
        return j * p + k
        
    def _iw(k, i):
        return k * n + i
    
    # --- Strassen Part (7 terms for 2x2 block) ---
    
    # r=0: m1 = (a00+a11)(b00+b11)
    r = 0
    U[_iu(0, 0), r] = 1
    U[_iu(1, 1), r] = 1
    V[_iv(0, 0), r] = 1
    V[_iv(1, 1), r] = 1
    W[_iw(0, 0), r] = 1
    W[_iw(1, 1), r] = 1
    
    # r=1: m2 = (a10+a11)b00
    r = 1
    U[_iu(1, 0), r] = 1
    U[_iu(1, 1), r] = 1
    V[_iv(0, 0), r] = 1
    W[_iw(0, 1), r] = 1
    W[_iw(1, 1), r] = -1
    
    # r=2: m3 = a00(b01-b11)
    r = 2
    U[_iu(0, 0), r] = 1
    V[_iv(0, 1), r] = 1
    V[_iv(1, 1), r] = -1
    W[_iw(1, 0), r] = 1
    W[_iw(1, 1), r] = 1
    
    # r=3: m4 = a11(b10-b00)
    r = 3
    U[_iu(1, 1), r] = 1
    V[_iv(1, 0), r] = 1
    V[_iv(0, 0), r] = -1
    W[_iw(0, 0), r] = 1
    W[_iw(0, 1), r] = 1
    
    # r=4: m5 = (a00+a01)b11
    r = 4
    U[_iu(0, 0), r] = 1
    U[_iu(0, 1), r] = 1
    V[_iv(1, 1), r] = 1
    W[_iw(0, 0), r] = -1
    W[_iw(1, 0), r] = 1
    
    # r=5: m6 = (a10-a00)(b00+b01)
    r = 5
    U[_iu(1, 0), r] = 1
    U[_iu(0, 0), r] = -1
    V[_iv(0, 0), r] = 1
    V[_iv(0, 1), r] = 1
    W[_iw(1, 1), r] = 1
    
    # r=6: m7 = (a01-a11)(b10+b11)
    r = 6
    U[_iu(0, 1), r] = 1
    U[_iu(1, 1), r] = -1
    V[_iv(1, 0), r] = 1
    V[_iv(1, 1), r] = 1
    W[_iw(0, 0), r] = 1
    
    # --- Border Terms (19 terms) ---
    # All (i,j,k) where max(i,j,k) >= 2, excluding those handled by Strassen
    border = [
        (2, 0, 0), (2, 0, 1), (2, 0, 2),
        (2, 1, 0), (2, 1, 1), (2, 1, 2),
        (2, 2, 0), (2, 2, 1), (2, 2, 2),
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
        (1, 2, 0), (1, 2, 1), (1, 2, 2),
        (0, 0, 2), (0, 1, 2),
        (1, 0, 2), (1, 1, 2)
    ]
    
    assert len(border) == 19
    
    col = 7
    for (i, j, k) in border:
        U[_iu(i, j), col] = 1
        V[_iv(j, k), col] = 1
        W[_iw(k, i), col] = 1
        col += 1
    
    assert col == R
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
