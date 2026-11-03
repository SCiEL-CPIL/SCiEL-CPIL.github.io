import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    
    # Helper to add a Strassen block
    # i_offset, j_offset, k_offset
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal c
        
        # M1 = (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
        U[i_off*3 + j_off][c] += 1
        U[(i_off+1)*3 + (j_off+1)][c] += 1
        V[j_off*5 + k_off][c] += 1
        V[(j_off+1)*5 + (k_off+1)][c] += 1
        W[k_off*2 + i_off][c] += 1
        W[(k_off+1)*2 + (i_off+1)][c] += 1
        c += 1
        
        # M2 = (A10+A11)*B00 -> C10(+1), C11(-1)
        U[(i_off+1)*3 + j_off][c] += 1
        U[(i_off+1)*3 + (j_off+1)][c] += 1
        V[j_off*5 + k_off][c] += 1
        W[k_off*2 + (i_off+1)][c] += 1
        W[(k_off+1)*2 + (i_off+1)][c] -= 1
        c += 1
        
        # M3 = A00*(B01-B11) -> C01(+1), C11(+1)
        U[i_off*3 + j_off][c] += 1
        V[(j_off)*5 + (k_off+1)][c] += 1
        V[(j_off+1)*5 + (k_off+1)][c] -= 1
        W[(k_off+1)*2 + i_off][c] += 1
        W[(k_off+1)*2 + (i_off+1)][c] += 1
        c += 1
        
        # M4 = A11*(B10-B00) -> C00(+1), C10(+1)
        U[(i_off+1)*3 + (j_off+1)][c] += 1
        V[(j_off+1)*5 + k_off][c] += 1
        V[j_off*5 + k_off][c] -= 1
        W[k_off*2 + i_off][c] += 1
        W[k_off*2 + (i_off+1)][c] += 1
        c += 1
        
        # M5 = (A00+A01)*B11 -> C00(-1), C01(+1)
        U[i_off*3 + j_off][c] += 1
        U[i_off*3 + (j_off+1)][c] += 1
        V[(j_off+1)*5 + (k_off+1)][c] += 1
        W[k_off*2 + i_off][c] -= 1
        W[(k_off+1)*2 + i_off][c] += 1
        c += 1
        
        # M6 = (A10-A00)*(B00+B01) -> C11(+1)
        U[(i_off+1)*3 + j_off][c] += 1
        U[i_off*3 + j_off][c] -= 1
        V[j_off*5 + k_off][c] += 1
        V[j_off*5 + (k_off+1)][c] += 1
        W[(k_off+1)*2 + (i_off+1)][c] += 1
        c += 1
        
        # M7 = (A01-A11)*(B10+B11) -> C00(+1)
        U[i_off*3 + (j_off+1)][c] += 1
        U[(i_off+1)*3 + (j_off+1)][c] -= 1
        V[(j_off+1)*5 + k_off][c] += 1
        V[(j_off+1)*5 + (k_off+1)][c] += 1
        W[k_off*2 + i_off][c] += 1
        c += 1

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    
    # Block 2: i in {0,1}, j in {1,2}, k in {2,3}
    add_strassen_block(0, 1, 2)
    
    # Remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if covered by Block 1
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue
                # Check if covered by Block 2
                if i in (0, 1) and j in (1, 2) and k in (2, 3):
                    continue
                
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
                
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
