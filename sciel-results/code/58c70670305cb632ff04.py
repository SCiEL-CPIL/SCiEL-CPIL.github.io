"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a Strassen term
    def add_strassen(i_off, j_off, k_off, col_start):
        # i_off: offset for i (0 or 1)
        # j_off: offset for j (0 or 1)
        # k_off: offset for k (0 or 2)
        
        # M1: (A00+A11)(B00+B11)
        c = col_start
        U[(i_off+0)*m + (j_off+0)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = 1
        V[(j_off+0)*p + (k_off+0)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off+0)*n + (i_off+0)][c] = 1
        W[(k_off+1)*n + (i_off+0)][c] = 1
        W[(k_off+0)*n + (i_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        c += 1

        # M2: (A10+A11)B00
        c = col_start + 1
        U[(i_off+1)*m + (j_off+0)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = 1
        V[(j_off+0)*p + (k_off+0)][c] = 1
        W[(k_off+0)*n + (i_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = -1
        c += 1

        # M3: A00(B01-B11)
        c = col_start + 2
        U[(i_off+0)*m + (j_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + (i_off+0)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        c += 1

        # M4: A11(B10-B00)
        c = col_start + 3
        U[(i_off+1)*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+0)][c] = -1
        W[(k_off+0)*n + (i_off+0)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        c += 1

        # M5: (A00+A01)B11
        c = col_start + 4
        U[(i_off+0)*m + (j_off+0)][c] = 1
        U[(i_off+0)*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off+0)*n + (i_off+0)][c] = -1
        W[(k_off+1)*n + (i_off+0)][c] = 1
        c += 1

        # M6: (A10-A00)(B00+B01)
        c = col_start + 5
        U[(i_off+1)*m + (j_off+0)][c] = 1
        U[(i_off+0)*m + (j_off+0)][c] = -1
        V[(j_off+0)*p + (k_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + (i_off+1)][c] = 1
        c += 1

        # M7: (A01-A11)(B10+B11)
        c = col_start + 6
        U[(i_off+0)*m + (j_off+1)][c] = 1
        U[(i_off+1)*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + (k_off+0)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off+0)*n + (i_off+0)][c] = -1
        W[(k_off+1)*n + (i_off+1)][c] = -1

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0, 0)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2, 7)
    
    # Block 3: j=2 (A_right, B_bottom)
    # A_right is 2x1 (i=0,1; j=2)
    # B_bottom is 1x4 (j=2; k=0,1,2,3)
    # C is 2x4 (i=0,1; k=0,1,2,3)
    # Terms: A[i,2] * B[2,k]
    # Rank 8 terms
    for i in range(n):
        for k in range(p):
            c = 14 + i * p + k
            U[i * m + 2][c] = 1
            V[2 * p + k][c] = 1
            W[k * n + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
