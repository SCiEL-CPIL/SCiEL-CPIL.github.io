import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 100. We aim for 92.
    # Block decomposition:
    # A: 4x5 -> [2x2, 2x2, 2x1] (rows 0-1, 2-3; cols 0-1, 2-3, 4)
    # B: 5x5 -> [2x2, 2x2, 2x1] (rows 0-1, 2-3, 4; cols 0-1, 2-3, 4)
    
    # Total Rank 92
    rank = 92
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # A indices: i in {0,1} (global row offset), j in {0,1} (global col offset)
    # B indices: j in {0,1} (global row offset), k in {0,1} (global col offset)
    # C indices: i in {0,1}, k in {0,1}
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        
        # M1 = (A00+A11)(B00+B11)
        U[i_off*2 + 0, col] += 1
        U[i_off*2 + 1, col] += 1
        V[j_off*2 + 0, col] += 1
        V[j_off*2 + 1, col] += 1
        # C00 (k=0, i=0) and C11 (k=1, i=1)
        W[k_off*2*n + i_off*2 + 0, col] += 1
        W[(k_off*2+1)*n + i_off*2 + 1, col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[i_off*2 + 1, col] += 1 # A10
        U[i_off*2 + 1, col] += 1 # A11 -- Wait, A10 is (i=1, j=0), A11 is (i=1, j=1).
        # U index: i*m + j.
        # A10: (i_off*2+1)*m + (j_off*2+0)
        # A11: (i_off*2+1)*m + (j_off*2+1)
        U[(i_off*2+1)*m + (j_off*2+0), col] += 1
        U[(i_off*2+1)*m + (j_off*2+1), col] += 1
        V[j_off*2 + 0, col] += 1 # B00
        # C10 (k=0, i=1)
        W[k_off*2*n + i_off*2 + 1, col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[i_off*2 + 0, col] += 1 # A00
        V[j_off*2 + 0, col] += 1 # B01? No.
        # B01 is (j=0, k=1). V index: j*p + k.
        # B01: (j_off*2+0)*p + (k_off*2+1)
        # B11: (j_off*2+1)*p + (k_off*2+1)
        U[(i_off*2+0)*m + (j_off*2+0), col] += 1
        V[(j_off*2+0)*p + (k_off*2+1), col] += 1
        V[(j_off*2+1)*p + (k_off*2+1), col] -= 1
        # C01 (k=1, i=0)
        W[(k_off*2+1)*n + i_off*2 + 0, col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        # A11: (i=1, j=1). B10: (j=1, k=0). B00: (j=0, k=0).
        U[(i_off*2+1)*m + (j_off*2+1), col] += 1
        V[(j_off*2+1)*p + (k_off*2+0), col] += 1
        V[(j_off*2+0)*p + (k_off*2+0), col] -= 1
        # C00 (k=0, i=0) and C10 (k=0, i=1)
        W[k_off*2*n + i_off*2 + 0, col] += 1
        W[k_off*2*n + i_off*2 + 1, col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        # A00: (0,0), A01: (0,1). B11: (1,1).
        U[(i_off*2+0)*m + (j_off*2+0), col] += 1
        U[(i_off*2+0)*m + (j_off*2+1), col] += 1
        V[(j_off*2+1)*p + (k_off*2+1), col] += 1
        # C01 (k=1, i=0) and C00 (k=0, i=0) with -1
        W[(k_off*2+1)*n + i_off*2 + 0, col] += 1
        W[k_off*2*n + i_off*2 + 0, col] -= 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10: (1,0), A00: (0,0). B00: (0,0), B01: (0,1).
        U[(i_off*2+1)*m + (j_off*2+0), col] += 1
        U[(i_off*2+0)*m + (j_off*2+0), col] -= 1
        V[(j_off*2+0)*p + (k_off*2+0), col] += 1
        V[(j_off*2+0)*p + (k_off*2+1), col] += 1
        # C11 (k=1, i=1)
        W[(k_off*2+1)*n + i_off*2 + 1, col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01: (0,1), A11: (1,1). B10: (1,0), B11: (1,1).
        U[(i_off*2+0)*m + (j_off*2+1), col] += 1
        U[(i_off*2+1)*m + (j_off*2+1), col] -= 1
        V[(j_off*2+1)*p + (k_off*2+0), col] += 1
        V[(j_off*2+1)*p + (k_off*2+1), col] += 1
        # C00 (k=0, i=0)
        W[k_off*2*n + i_off*2 + 0, col] += 1
        col += 1

    def add_schoolbook_block(i_off, j_off, k_off, h, w, d):
        # h: rows of A block, w: cols of A block (rows of B block), d: cols of B block
        # A is h x w, B is w x d, C is h x d
        nonlocal col
        for ii in range(h):
            for ww in range(w):
                for dd in range(d):
                    U[(i_off+ii)*m + (j_off+ww), col] = 1
                    V[(j_off+ww)*p + (k_off+dd), col] = 1
                    W[(k_off+dd)*n + (i_off+ii), col] = 1
                    col += 1

    # 8 blocks of 2x2x2
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            for k_off in [0, 2]:
                add_strassen_2x2x2(i_off, j_off, k_off)
                
    # j=4 terms (A is 2x1, B is 1x2 or 1x1)
    # i in {0, 2} (offsets for 2 rows)
    # j=4 (offset for 1 col)
    for i_off in [0, 2]:
        # k=0,1 (B is 1x2)
        add_schoolbook_block(i_off, 4, 0, 2, 1, 2)
        add_schoolbook_block(i_off, 4, 2, 2, 1, 2)
        # k=2 (B is 1x1) -- Wait, B cols are 0-4.
        # k_off=0 covers B cols 0,1.
        # k_off=2 covers B cols 2,3.
        # k_off=4 covers B col 4.
        add_schoolbook_block(i_off, 4, 4, 2, 1, 1)
        
    # k=4 terms (A is 2x2, B is 2x1)
    # i in {0, 2}, j in {0, 2}
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            add_schoolbook_block(i_off, j_off, 4, 2, 2, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
