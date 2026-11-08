<<<<<<< SEARCH
import time
import random

DIMENSIONS = [3, 3, 4]

_DEADLINE = 120.0
_START = None


def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)
=======
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

_DEADLINE = 115.0
_START = None


def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)


def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (each product term A_ij * B_jk contributes to C_ik)
    # Actually the tensor is: for each (i,j,k), the coefficient is 1
    # So we check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
>>>>>>> REPLACE

