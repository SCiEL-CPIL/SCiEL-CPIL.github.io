import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 8]
DEADLINE = 140  # seconds, safety margin under 150

def _all_ones_tensor(n, m, p):
    return np.ones((n, m, p))

def _decompose_block_rank1(n, m, p, U, V, W):
    # U: (n, R), V: (m, R), W: (p, R)
    # returns the tensor
    R = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(R):
        T += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
    return T

def _strassen_2x2x2():
    # Returns (U,V,W) each of shape (2,7) for the 2x2x2 all-ones tensor
    # Strassen: 7 products
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (b-d)(e+g)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    # U[i, r] = coeff of A_{i,j} in M_r ... 
    # This is getting complicated. Let me just use the standard Strassen rank-7 decomposition.
    pass
