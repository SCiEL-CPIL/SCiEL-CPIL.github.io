import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    current_rank = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal current_rank
        # Term 1: (A11+A22)(B11+B22)
        U[(i_off+0)*m + (j_off+0)][current_rank] += 1
        U[(i_off+1)*m + (j_off+1)][current_rank] += 1
        V[(j_off+0)*p + (k_off+0)][current_rank] += 1
        V[(j_off+1)*p + (k_off+1)][current_rank] += 1
        W[(k_off+0)*n + (i_off+0)][current_rank] += 1
        W[(k_off+1)*n + (i_off+1)][current_rank] += 1
        current_rank += 1
        
        # Term 2: (A21+A22)B11
        U[(i_off+1)*m + (j_off+0)][current_rank] += 1
        U[(i_off+1)*m + (j_off+1)][current_rank] += 1
        V[(j_off+0)*p + (k_off+0)][current_rank] += 1
        W[(k_off+0)*n + (i_off+1)][current_rank] += 1
        current_rank += 1
        
        # Term 3: A11(B12-B22)
        U[(i_off+0)*m + (j_off+0)][current_rank] += 1
        V[(j_off+0)*p + (k_off+1)][current_rank] += 1
        V[(j_off+1)*p + (k_off+1)][current_rank] -= 1
        W[(k_off+1)*n + (i_off+0)][current_rank] += 1
        current_rank += 1
        
        # Term 4: A22(B21-B11)
        U[(i_off+1)*m + (j_off+1)][current_rank] += 1
        V[(j_off+1)*p + (k_off+0)][current_rank] += 1
        V[(j_off+0)*p + (k_off+0)][current_rank] -= 1
        W[(k_off+0)*n + (i_off+1)][current_rank] += 1
        current_rank += 1
        
        # Term 5: (A11+A12)B22
        U[(i_off+0)*m + (j_off+0)][current_rank] += 1
        U[(i_off+0)*m + (j_off+1)][current_rank] += 1
        V[(j_off+1)*p + (k_off+1)][current_rank] += 1
        W[(k_off+1)*n + (i_off+0)][current_rank] += 1
        current_rank += 1
        
        # Term 6: (A21-A11)(B11+B12)
        U[(i_off+1)*m + (j_off+0)][current_rank] += 1
        U[(i_off+0)*m + (j_off+0)][current_rank] -= 1
        V[(j_off+0)*p + (k_off+0)][current_rank] += 1
        V[(j_off+0)*p + (k_off+1)][current_rank] += 1
        W[(k_off+0)*n + (i_off+1)][current_rank] += 1
        current_rank += 1
        
        # Term 7: (A12-A22)(B21+B22)
        U[(i_off+0)*m + (j_off+1)][current_rank] += 1
        U[(i_off+1)*m + (j_off+1)][current_rank] -= 1
        V[(j_off+1)*p + (k_off+0)][current_rank] += 1
        V[(j_off+1)*p + (k_off+1)][current_rank] += 1
        W[(k_off+1)*n + (i_off+0)][current_rank] += 1
        current_rank += 1

    def add_schoolbook_1x2x2(i_off, j_off, k_off):
        nonlocal current_rank
        # Term 1: A00 B00 -> C00
        U[i_off*m + j_off][current_rank] += 1
        V[j_off*p + k_off][current_rank] += 1
        W[k_off*n + i_off][current_rank] += 1
        current_rank += 1
        
        # Term 2: A01 B10 -> C00
        U[i_off*m + j_off+1][current_rank] += 1
        V[j_off+1*p + k_off][current_rank] += 1
        W[k_off*n + i_off][current_rank] += 1
        current_rank += 1
        
        # Term 3: A00 B01 -> C01
        U[i_off*m + j_off][current_rank] += 1
        V[j_off*p + k_off+1][current_rank] += 1
        W[k_off+1*n + i_off][current_rank] += 1
        current_rank += 1
        
        # Term 4: A01 B11 -> C01
        U[i_off*m + j_off+1][current_rank] += 1
        V[j_off+1*p + k_off+1][current_rank] += 1
        W[k_off+1*n + i_off][current_rank] += 1
        current_rank += 1

    # Apply blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 4, 4)
    
    add_schoolbook_1x2x2(2, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2)
    add_schoolbook_1x2x2(2, 4, 4)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
