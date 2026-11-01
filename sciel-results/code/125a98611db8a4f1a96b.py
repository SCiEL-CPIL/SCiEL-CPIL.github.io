# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    The 7 rank-1 terms have coefficients in {-1, 0, 1}.
    U, V, W shapes: (4, 7)
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

def _construct_rank_56():
    """
    Construct the rank-56 decomposition by using 8 independent Strassen blocks.
    """
    n, m, p = 2, 2, 16
    U_core, V_core, W_core = _strassen_2x2()
    num_pairs = p // 2
    rank = 7 * num_pairs
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        col_offset = t * 7
        for r in range(7):
            # U is shared across all blocks (same A)
            U[:, col_offset + r] = U_core[:, r]
            
            # V maps B[j, k] to the block t
            # V index: j*p + k
            # For block t, k is 2*t and 2*t+1
            # V_core row 0 corresponds to B[0, 0] (local)
            # V_core row 1 corresponds to B[0, 1] (local) -- Wait, Strassen V is 4x7 for 2x2x2.
            # V_core[0] is for B[0,0], V_core[1] for B[0,1], V_core[2] for B[1,0], V_core[3] for B[1,1].
            # In the 2x16 matrix, B[0, 2t] is local B[0,0], B[0, 2t+1] is local B[0,1].
            # B[1, 2t] is local B[1,0], B[1, 2t+1] is local B[1,1].
            
            V[0 * p + 2*t, col_offset + r] = V_core[0, r]
            V[0 * p + 2*t + 1, col_offset + r] = V_core[1, r]
            V[1 * p + 2*t, col_offset + r] = V_core[2, r]
            V[1 * p + 2*t + 1, col_offset + r] = V_core[3, r]
            
            # W maps C[i, k]
            # W index: k*n + i
            # For block t, k is 2*t and 2*t+1.
            # W_core row 0 corresponds to C[0, 0] (local)
            # W_core row 1 corresponds to C[0, 1] (local)
            # W_core row 2 corresponds to C[1, 0] (local)
            # W_core row 3 corresponds to C[1, 1] (local)
            
            # W[k*n + i, r]
            # k=2t, i=0 -> W[2t*2 + 0] = W[4t]
            # k=2t, i=1 -> W[2t*2 + 1] = W[4t+1]
            # k=2t+1, i=0 -> W[(2t+1)*2 + 0] = W[4t+2]
            # k=2t+1, i=1 -> W[(2t+1)*2 + 1] = W[4t+3]
            
            W[4*t, col_offset + r] = W_core[0, r]
            W[4*t + 1, col_offset + r] = W_core[1, r]
            W[4*t + 2, col_offset + r] = W_core[2, r]
            W[4*t + 3, col_offset + r] = W_core[3, r]
            
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    """
    U, V, W = _construct_rank_56()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
