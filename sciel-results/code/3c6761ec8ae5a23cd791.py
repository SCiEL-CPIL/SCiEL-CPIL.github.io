import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses the standard schoolbook algorithm which guarantees correctness.
    Rank = n * m * p = 60.
    """
    n, m, p = DIMENSIONS
    
    # Check deadline (though schoolbook is instant)
    if time.time() > _DEADLINE:
        return _schoolbook_fallback(n, m, p)
    
    return _schoolbook_fallback(n, m, p)

def _schoolbook_fallback(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
