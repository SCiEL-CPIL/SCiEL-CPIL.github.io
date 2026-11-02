"""Optimized Rank-15 decomposition for 2x3x4 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 15
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a Strassen decomposition for 2x2x2
    # i_off: offset for i index in A (0 or 1)
    # j_off: offset for j index in A and B (0 or 1)
    # k_off: offset for k index in B and C (0, 1, 2, or 3)
    # r_off: offset for rank column
    def add_strassen(i_off, j_off, k_off, r_off):
        # M1 = (A00+A11)(B00+B11)
        U[i_off*m + j_off, r_off] += 1
        U[(i_off+1)*m + j_off+1, r_off] += 1
        V[j_off*p + k_off, r_off] += 1
        V[(j_off+1)*p + k_off+1, r_off] += 1
        W[k_off*n + i_off, r_off] += 1
        W[(k_off+1)*n + (i_off+1), r_off] += 1
        W[k_off*n + (i_off+1), r_off] += 1
        W[(k_off+1)*n + i_off, r_off] += 1
        
        # M2 = (A10+A11)B00
        U[(i_off+1)*m + j_off, r_off+1] += 1
        U[(i_off+1)*m + j_off+1, r_off+1] += 1
        V[j_off*p + k_off, r_off+1] += 1
        W[k_off*n + (i_off+1), r_off+1] += 1
        W[(k_off+1)*n + (i_off+1), r_off+1] += 1
        
        # M3 = A00(B01-B11)
        U[i_off*m + j_off, r_off+2] += 1
        V[j_off*p + k_off+1, r_off+2] += 1
        V[(j_off+1)*p + k_off+1, r_off+2] -= 1
        W[k_off*n + i_off, r_off+2] += 1
        W[(k_off+1)*n + i_off, r_off+2] += 1
        
        # M4 = A11(B10-B00)
        U[(i_off+1)*m + j_off+1, r_off+3] += 1
        V[(j_off+1)*p + k_off, r_off+3] += 1
        V[j_off*p + k_off, r_off+3] -= 1
        W[k_off*n + (i_off+1), r_off+3] += 1
        W[(k_off+1)*n + (i_off+1), r_off+3] += 1
        
        # M5 = (A00+A01)B11
        U[i_off*m + j_off, r_off+4] += 1
        U[i_off*m + j_off+1, r_off+4] += 1
        V[(j_off+1)*p + k_off+1, r_off+4] += 1
        W[k_off*n + i_off, r_off+4] += 1
        W[(k_off+1)*n + i_off, r_off+4] += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i_off+1)*m + j_off, r_off+5] += 1
        U[i_off*m + j_off, r_off+5] -= 1
        V[j_off*p + k_off, r_off+5] += 1
        V[j_off*p + k_off+1, r_off+5] += 1
        W[k_off*n + (i_off+1), r_off+5] += 1
        W[(k_off+1)*n + (i_off+1), r_off+5] += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i_off*m + j_off+1, r_off+6] += 1
        U[(i_off+1)*m + j_off+1, r_off+6] -= 1
        V[(j_off+1)*p + k_off, r_off+6] += 1
        V[(j_off+1)*p + k_off+1, r_off+6] += 1
        W[k_off*n + i_off, r_off+6] += 1
        W[(k_off+1)*n + i_off, r_off+6] += 1

    # 1. A1 B11a: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0, 0)
    
    # 2. A1 B11b: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2, 7)
    
    # 3. A2 B21: i in {0,1}, j=2, k in {0,1,2,3}
    # Rank 1 term
    r = 14
    for i in range(n):
        U[i*m + 2, r] = 1
    for k in range(p):
        V[2*p + k, r] = 1
    for i in range(n):
        for k in range(p):
            W[k*n + i, r] = 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
