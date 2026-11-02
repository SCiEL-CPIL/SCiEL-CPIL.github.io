"""Matrix multiplication 2x2 x 2x16 via Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    independently to each of the 8 column-pairs of the 2x16 matrix.
    Total rank = 7 * 8 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    U_core, V_core, W_core = _strassen_2x2()
    
    num_pairs = p // 2  # 8 pairs
    rank = 7 * num_pairs  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Vectorized construction
    # U is the same for all blocks
    # U[:, t*7 : (t+1)*7] = U_core
    for t in range(num_pairs):
        if _time_remaining() < 1.0:
            break
        U[:, t*7:(t+1)*7] = U_core
        
        # V indices: j*p + k
        # For block t, k is 2t and 2t+1
        # j=0: indices 2t, 2t+1
        # j=1: indices p+2t, p+2t+1
        V[2*t, t*7:(t+1)*7] = V_core[0, :]
        V[2*t+1, t*7:(t+1)*7] = V_core[1, :]
        V[p+2*t, t*7:(t+1)*7] = V_core[2, :]
        V[p+2*t+1, t*7:(t+1)*7] = V_core[3, :]
        
        # W indices: k*n + i
        # For block t, k is 2t and 2t+1
        # k=2t: indices 4t, 4t+1
        # k=2t+1: indices 4t+2, 4t+3
        # W_core rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W[4t] = W_core[0] (i=0, k=2t)
        # W[4t+1] = W_core[2] (i=1, k=2t)
        # W[4t+2] = W_core[1] (i=0, k=2t+1)
        # W[4t+3] = W_core[3] (i=1, k=2t+1)
        W[4*t, t*7:(t+1)*7] = W_core[0, :]
        W[4*t+1, t*7:(t+1)*7] = W_core[2, :]
        W[4*t+2, t*7:(t+1)*7] = W_core[1, :]
        W[4*t+3, t*7:(t+1)*7] = W_core[3, :]
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
