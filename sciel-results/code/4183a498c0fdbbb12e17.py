"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-13 decomposition for 2x2 * 2x5 matrix multiplication.
    Uses Strassen (rank 7) for the first two columns and rank-2 per column for the rest.
    """
    n, m, p = 2, 2, 5
    rank = 13
    
    U = np.zeros((n * m, rank), dtype=int)  # (4, 13)
    V = np.zeros((m * p, rank), dtype=int)  # (10, 13)
    W = np.zeros((p * n, rank), dtype=int)  # (10, 13)
    
    # Strassen terms for columns 0,1: c = 0..6
    # U indices: i*2+j, V indices: j*5+k, W indices: k*2+i
    
    # M0 = (A00+A11)(B00+B11) -> C[0,0], C[1,1]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M1 = (A10+A11)B00 -> C[1,0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M2 = A00(B10-B11) -> C[0,1], C[1,1]
    U[0, 2] = 1
    V[5, 2] = 1; V[6, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M3 = A11(B01-B00) -> C[0,0], C[1,0]
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M4 = (A00+A01)B11 -> C[0,0], C[0,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[6, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M5 = (A10-A00)(B00+B01) -> C[1,1]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = -1
    
    # M6 = (A01-A11)(B10+B11) -> C[0,0], C[1,1]
    U[1, 6] = 1; U[3, 6] = -1
    V[5, 6] = 1; V[6, 6] = 1
    W[0, 6] = 1; W[3, 6] = -1
    
    # Schoolbook terms for columns k=2,3,4: c = 7..12
    for k in range(2, 5):
        c1 = 7 + (k - 2) * 2
        c2 = c1 + 1
        # c1: (A00+A10)*B0k -> C[0,k], C[1,k]
        U[0, c1] = 1; U[2, c1] = 1
        V[k, c1] = 1
        W[k * 2, c1] = 1; W[k * 2 + 1, c1] = 1
        # c2: (A01+A11)*B1k -> C[0,k], C[1,k]
        U[1, c2] = 1; U[3, c2] = 1
        V[5 + k, c2] = 1
        W[k * 2, c2] = 1; W[k * 2 + 1, c2] = 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
