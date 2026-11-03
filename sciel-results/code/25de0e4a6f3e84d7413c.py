import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 4 matrix.
    Returns a dictionary with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 4, 4
    
    # Rank 32 solution:
    # 2x4x4 part: 4 blocks of 2x2x2 Strassen (rank 7 each) -> 28
    # 1x4x4 part: 4 trivial terms -> 4
    # Total rank 32
    rank = 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2
    # i_base, j_base, k_base are the starting indices for i, j, k
    def add_strassen(i_base, j_base, k_base, start_col):
        # M1 = (A00+A11)(B00+B11)
        U[i_base*m + j_base, start_col] = 1
        U[(i_base+1)*m + (j_base+1), start_col] = 1
        V[j_base*p + k_base, start_col] = 1
        V[(j_base+1)*p + (k_base+1), start_col] = 1
        W[k_base*n + i_base, start_col] = 1
        W[(k_base+1)*n + (i_base+1), start_col] = 1
        
        # M2 = (A10+A11)B00
        U[(i_base+1)*m + j_base, start_col+1] = 1
        U[(i_base+1)*m + (j_base+1), start_col+1] = 1
        V[j_base*p + k_base, start_col+1] = 1
        W[k_base*n + (i_base+1), start_col+1] = 1
        W[(k_base+1)*n + (i_base+1), start_col+1] = -1
        
        # M3 = A00(B10-B00)
        U[i_base*m + j_base, start_col+2] = 1
        V[(j_base+1)*p + k_base, start_col+2] = 1
        V[j_base*p + k_base, start_col+2] = -1
        W[(k_base+1)*n + i_base, start_col+2] = 1
        W[(k_base+1)*n + (i_base+1), start_col+2] = 1
        
        # M4 = A11(B01-B11)
        U[(i_base+1)*m + (j_base+1), start_col+3] = 1
        V[j_base*p + (k_base+1), start_col+3] = 1
        V[(j_base+1)*p + (k_base+1), start_col+3] = -1
        W[k_base*n + i_base, start_col+3] = 1
        W[k_base*n + (i_base+1), start_col+3] = 1
        
        # M5 = (A00+A01)B11
        U[i_base*m + j_base, start_col+4] = 1
        U[i_base*m + (j_base+1), start_col+4] = 1
        V[(j_base+1)*p + (k_base+1), start_col+4] = 1
        W[k_base*n + i_base, start_col+4] = -1
        W[(k_base+1)*n + i_base, start_col+4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i_base+1)*m + j_base, start_col+5] = 1
        U[i_base*m + j_base, start_col+5] = -1
        V[j_base*p + k_base, start_col+5] = 1
        V[j_base*p + (k_base+1), start_col+5] = 1
        W[(k_base+1)*n + (i_base+1), start_col+5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i_base*m + (j_base+1), start_col+6] = 1
        U[(i_base+1)*m + (j_base+1), start_col+6] = -1
        V[(j_base+1)*p + k_base, start_col+6] = 1
        V[(j_base+1)*p + (k_base+1), start_col+6] = 1
        W[k_base*n + i_base, start_col+6] = 1

    # Part 1: Rows 0,1 of A (2x4x4)
    # Split into 4 blocks of 2x2x2
    # j_1 in {0, 1}, k_1 in {0, 1}
    for j_1 in range(2):
        for k_1 in range(2):
            i_base = 0
            j_base = j_1 * 2
            k_base = k_1 * 2
            add_strassen(i_base, j_base, k_base, col)
            col += 7
            
    # Part 2: Row 2 of A (1x4x4)
    # 4 trivial terms
    for r in range(4):
        U[2*m + r, col] = 1
        for k in range(4):
            V[r*p + k, col] = 1
            W[k*n + 2, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
