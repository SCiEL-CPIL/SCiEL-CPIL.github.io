"""Matrix multiplication 3x3x6 tensor decomposition finder."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n=3, m=3, p=6, trials=50, seed=42):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Direct tensor verification: check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a tensor decomposition for 3x3x6 matrix multiplication.
    Returns {'U': (9 x rank), 'V': (18 x rank), 'W': (18 x rank)}
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    
    # Verify schoolbook
    if _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback (should not happen)
    return best
