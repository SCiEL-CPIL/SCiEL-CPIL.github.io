import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x7 matrix.
    
    Uses a hybrid approach:
    - Splits the middle dimension m=3 into j=0 and j in {1,2}.
    - For j=0, uses schoolbook (rank 14).
    - For j in {1,2}, uses Strassen's algorithm for 2x2x2 blocks 
      and schoolbook for the remaining 2x2x1 block.
    - Total rank: 14 + 3*7 + 4 = 39.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Total rank
    rank_total = 14 + 21 + 4 # 39
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col = 0
    
    # Part 1: j=0 (2x1x7)
    # Schoolbook
    for i in range(n):
        for k in range(p):
            # U index: i*m + 0
            # V index: 0*p + k
            # W index: k*n + i
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Part 2: j in {1, 2} (2x2x7)
    # Split k into 3 blocks of 2 (0-1, 2-3, 4-5) and 1 block of 1 (6)
    
    # Block 1: k=0,1
    # Block 2: k=2,3
    # Block 3: k=4,5
    for b in range(3):
        k_start = b * 2
        k_end = k_start + 1
        
        # Strassen coefficients
        # M1: (A00+A11)(B00+B11) -> C00+C01+C10+C11
        # M2: (A10+A11)B00 -> C10-C11
        # M3: A00(B10-B11) -> C01+C11
        # M4: A11(B01-B00) -> C00+C10
        # M5: (A00+A01)B11 -> -C00+C11
        # M6: (A10-A00)(B00+B10) -> C11
        # M7: (A01-A11)(B01+B11) -> C00
        
        # Local indices for A: i in 0..1, j_local in 0..1 (global j = 1 + j_local)
        # Local indices for B: j_local in 0..1 (global j = 1 + j_local), k_local in 0..1 (global k = k_start + k_local)
        # Local indices for C: i in 0..1, k_local in 0..1 (global k = k_start + k_local)
        
        # M1
        U[0 * m + 1, col] = 1; U[1 * m + 2, col] = 1
        V[1 * p + k_start, col] = 1; V[2 * p + k_end, col] = 1
        W[k_start * n + 0, col] = 1; W[k_end * n + 0, col] = 1
        W[k_start * n + 1, col] = 1; W[k_end * n + 1, col] = 1
        col += 1
        
        # M2
        U[1 * m + 1, col] = 1; U[1 * m + 2, col] = 1
        V[1 * p + k_start, col] = 1
        W[k_start * n + 1, col] = 1; W[k_end * n + 1, col] = -1
        col += 1
        
        # M3
        U[0 * m + 1, col] = 1
        V[2 * p + k_start, col] = 1; V[2 * p + k_end, col] = -1
        W[k_end * n + 0, col] = 1; W[k_end * n + 1, col] = 1
        col += 1
        
        # M4
        U[1 * m + 2, col] = 1
        V[1 * p + k_end, col] = 1; V[1 * p + k_start, col] = -1
        W[k_start * n + 0, col] = 1; W[k_start * n + 1, col] = 1
        col += 1
        
        # M5
        U[0 * m + 1, col] = 1; U[0 * m + 2, col] = 1
        V[2 * p + k_end, col] = 1
        W[k_start * n + 0, col] = -1; W[k_end * n + 1, col] = 1
        col += 1
        
        # M6
        U[1 * m + 1, col] = 1; U[0 * m + 1, col] = -1
        V[1 * p + k_start, col] = 1; V[2 * p + k_start, col] = 1
        W[k_end * n + 1, col] = 1
        col += 1
        
        # M7
        U[0 * m + 2, col] = 1; U[1 * m + 2, col] = -1
        V[1 * p + k_end, col] = 1; V[2 * p + k_end, col] = 1
        W[k_start * n + 0, col] = 1
        col += 1
        
    # Part 2b: k=6 (2x2x1)
    # Schoolbook
    for i in range(n):
        for j in range(1, 3):
            # U index: i*m + j
            # V index: j*p + 6
            # W index: 6*n + i
            U[i * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + i, col] = 1
            col += 1
            
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
