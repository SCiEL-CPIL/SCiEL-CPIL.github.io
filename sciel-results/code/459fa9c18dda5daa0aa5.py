import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 6 blocks of 2x2x2 (Strassen, rank 7)
    # 1 block of 2x4x1 (rank 4)
    # 1 block of 1x4x6 (rank 4)
    # 1 block of 1x4x1 (rank 1)
    # Total rank = 6*7 + 4 + 4 + 1 = 51
    
    rank = 51
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen helper for 2x2x2 block
    # Local indices i' in {0,1}, j' in {0,1}, k' in {0,1}
    # Global offsets i_off, j_off, k_off
    def add_strassen_block(i_off, j_off, k_off, col):
        # Term 1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        U[i_off*m + j_off, col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[j_off*p + k_off, col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[k_off*n + i_off, col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # Term 2: (A10+A11)B00 -> C10(+1)
        U[(i_off+1)*m + j_off, col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[j_off*p + k_off, col] = 1
        W[k_off*n + (i_off+1), col] = 1
        col += 1
        
        # Term 3: A00 B10 -> C11(-1)
        U[i_off*m + j_off, col] = 1
        V[(j_off+1)*p + k_off, col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # Term 4: A11 B11 -> C00(-1), C10(+1)
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[k_off*n + i_off, col] = -1
        W[k_off*n + (i_off+1), col] = 1
        col += 1
        
        # Term 5: (A00+A01)B11 -> C01(+1)
        U[i_off*m + j_off, col] = 1
        U[i_off*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + i_off, col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B01) -> C00(+1)
        U[(i_off+1)*m + j_off, col] = 1
        U[i_off*m + j_off, col] = -1
        V[j_off*p + k_off, col] = 1
        V[j_off*p + (k_off+1), col] = 1
        W[k_off*n + i_off, col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B10+B11) -> C01(+1), C11(+1)
        U[i_off*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + k_off, col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + i_off, col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        return col

    # 6 Strassen blocks
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Blocks:
    # 1. i={0,1}, j={0,1}, k={0,1}
    col = add_strassen_block(0, 0, 0, col)
    # 2. i={0,1}, j={2,3}, k={0,1}
    col = add_strassen_block(0, 2, 0, col)
    # 3. i={0,1}, j={0,1}, k={2,3}
    col = add_strassen_block(0, 0, 2, col)
    # 4. i={0,1}, j={2,3}, k={2,3}
    col = add_strassen_block(0, 2, 2, col)
    # 5. i={0,1}, j={0,1}, k={4,5}
    col = add_strassen_block(0, 0, 4, col)
    # 6. i={0,1}, j={2,3}, k={4,5}
    col = add_strassen_block(0, 2, 4, col)
    
    # Block 2: i in {0,1}, j in {0,1,2,3}, k=6. Rank 4.
    # Terms r=0..3 corresponding to j=r
    for j in range(4):
        # U[i, j, r] = 1 if j==r
        U[0*m + j, col] = 1
        U[1*m + j, col] = 1
        # V[j, 6, r] = 1
        V[j*p + 6, col] = 1
        # W[6, i, r] = 1
        W[6*n + 0, col] = 1
        W[6*n + 1, col] = 1
        col += 1
        
    # Block 3: i=2, j in {0,1,2,3}, k in {0,1,2,3,4,5}. Rank 4.
    # Terms r=0..3 corresponding to j=r
    for j in range(4):
        # U[2, j, r] = 1
        U[2*m + j, col] = 1
        # V[j, k, r] = 1 for k in 0..5
        for k in range(6):
            V[j*p + k, col] = 1
        # W[k, 2, r] = 1 for k in 0..5
        for k in range(6):
            W[k*n + 2, col] = 1
        col += 1
        
    # Block 4: i=2, j in {0,1,2,3}, k=6. Rank 1.
    # U[2, j, 0] = 1
    for j in range(4):
        U[2*m + j, col] = 1
    # V[j, 6, 0] = 1
    for j in range(4):
        V[j*p + 6, col] = 1
    # W[6, 2, 0] = 1
    W[6*n + 2, col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
