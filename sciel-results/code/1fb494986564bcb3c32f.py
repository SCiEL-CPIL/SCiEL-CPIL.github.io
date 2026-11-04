import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank = 50
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen block for 2x2x2
    # i_offset, j_offset, k_offset
    # A is 2x2, B is 2x2
    # U rows: (i_offset+i)*m + (j_offset+j)
    # V rows: (j_offset+j)*p + (k_offset+k)
    # W rows: (k_offset+k)*n + (i_offset+i)
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1 = (a+d)(e+h)
        U[i_off*m + j_off, col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off)*p + k_off, col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[k_off*n + i_off, col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[i_off*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off)*p + k_off, col] = 1
        W[k_off*n + (i_off+1), col] = 1
        col += 1
        
        # M3 = a(f-h)
        U[i_off*m + j_off, col] = 1
        V[(j_off)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + i_off, col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + k_off, col] = 1
        V[(j_off)*p + k_off, col] = -1
        W[k_off*n + (i_off+1), col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[i_off*m + j_off, col] = 1
        U[i_off*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + i_off, col] = -1
        W[(k_off+1)*n + (i_off+1), col] = 1 # Wait, C11 has M5? No.
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        # So M5 contributes to C01 (1) and C00 (-1).
        # W index is (k, i).
        # C01 is i=0, k=1. W[1*n+0]
        # C00 is i=0, k=0. W[0*n+0]
        W[(k_off+1)*n + i_off, col] = 1
        W[k_off*n + i_off, col] = -1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[(i_off+1)*m + (j_off+1), col] = 1
        U[i_off*m + j_off, col] = -1
        V[(j_off)*p + k_off, col] = 1
        V[(j_off)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[i_off*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + k_off, col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[k_off*n + i_off, col] = 1
        W[(k_off+1)*n + i_off, col] = 1
        col += 1
        
    def add_schoolbook_2x1x1(i_off, j_off, k_off):
        # A is 2x1, B is 1x1
        # i in 0..1, j=0, k=0
        nonlocal col
        for i in range(2):
            U[(i_off+i)*m + (j_off), col] = 1
            V[(j_off)*p + (k_off), col] = 1
            W[(k_off)*n + (i_off+i), col] = 1
            col += 1

    def add_schoolbook_1x1x1(i_off, j_off, k_off):
        # A is 1x1, B is 1x1
        nonlocal col
        U[(i_off)*m + (j_off), col] = 1
        V[(j_off)*p + (k_off), col] = 1
        W[(k_off)*n + (i_off), col] = 1
        col += 1

    # Part 1: A1 (2x2) * B1 (2x9)
    # i_off = 0, j_off = 0
    # k pairs: (0,1), (2,3), (4,5), (6,7), (8)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 0, 6)
    # Single k=8
    # A1 is 2x2. B1 col 8 is 2x1.
    # So we need 2x2x1 decomposition.
    # Schoolbook for 2x2x1 is 4 terms.
    # i in 0..1, j in 0..1, k=0 (offset 8)
    for i in range(2):
        for j in range(2):
            U[i*m + j, col] = 1
            V[j*p + 8, col] = 1
            W[8*n + i, col] = 1
            col += 1

    # Part 2: A2 (2x1) * B2 (1x9)
    # i_off = 0, j_off = 2
    # k in 0..8
    # Schoolbook 2x1x9
    for k in range(9):
        for i in range(2):
            U[i*m + 2, col] = 1
            V[2*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
