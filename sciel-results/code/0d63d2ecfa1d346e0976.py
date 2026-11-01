import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    total_rank = 46
    
    # U: (n*m, rank) = (10, 46)
    # V: (m*p, rank) = (25, 46)
    # W: (p*n, rank) = (10, 46)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_block_strassen(i_off, j_off, k_off):
        nonlocal col
        # A indices: (i_off + 0, j_off + 0), (i_off + 0, j_off + 1), (i_off + 1, j_off + 0), (i_off + 1, j_off + 1)
        # B indices: (j_off + 0, k_off + 0), (j_off + 0, k_off + 1), (j_off + 1, k_off + 0), (j_off + 1, k_off + 1)
        # C indices: (i_off + 0, k_off + 0), (i_off + 0, k_off + 1), (i_off + 1, k_off + 0), (i_off + 1, k_off + 1)
        
        # Local indices 0,1 for i, j, k
        # A local (0,0) -> Global (i_off, j_off)
        # A local (0,1) -> Global (i_off, j_off+1)
        # A local (1,0) -> Global (i_off+1, j_off)
        # A local (1,1) -> Global (i_off+1, j_off+1)
        
        # B local (0,0) -> Global (j_off, k_off)
        # B local (0,1) -> Global (j_off, k_off+1)
        # B local (1,0) -> Global (j_off+1, k_off)
        # B local (1,1) -> Global (j_off+1, k_off+1)
        
        # C local (0,0) -> Global (i_off, k_off) -> W index k_off*n + i_off
        # C local (0,1) -> Global (i_off, k_off+1) -> W index (k_off+1)*n + i_off
        # C local (1,0) -> Global (i_off+1, k_off) -> W index k_off*n + (i_off+1)
        # C local (1,1) -> Global (i_off+1, k_off+1) -> W index (k_off+1)*n + (i_off+1)
        
        # M1: (a+d)(e+h)
        U[i_off*m + j_off][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[j_off*p + k_off][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[k_off*n + i_off][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M2: (c+d)e
        U[(i_off+1)*m + j_off][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[j_off*p + k_off][col] += 1
        W[k_off*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] -= 1
        col += 1
        
        # M3: a(f-h)
        U[i_off*m + j_off][col] += 1
        V[j_off*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] -= 1
        W[(k_off+1)*n + i_off][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M4: d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + k_off][col] += 1
        V[j_off*p + k_off][col] -= 1
        W[k_off*n + i_off][col] += 1
        W[k_off*n + (i_off+1)][col] += 1
        col += 1
        
        # M5: (a+b)h
        U[i_off*m + j_off][col] += 1
        U[i_off*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + i_off][col] += 1
        W[k_off*n + i_off][col] -= 1
        col += 1
        
        # M6: (c-a)(e+f)
        U[(i_off+1)*m + j_off][col] += 1
        U[i_off*m + j_off][col] -= 1
        V[j_off*p + k_off][col] += 1
        V[j_off*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + i_off][col] -= 1
        col += 1
        
        # M7: (b-d)(g+h)
        U[i_off*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] -= 1
        V[(j_off+1)*p + k_off][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[k_off*n + i_off][col] += 1
        col += 1

    def add_block_schoolbook_2x1x1x2(i_off, j_off, k_off):
        # A: 2x1 (i_off, i_off+1) x (j_off)
        # B: 1x2 (j_off) x (k_off, k_off+1)
        # C: 2x2
        nonlocal col
        # Terms: A[i,0] * B[0,k]
        # 4 terms
        for ii in range(2):
            for kk in range(2):
                U[(i_off+ii)*m + j_off][col] += 1
                V[j_off*p + (k_off+kk)][col] += 1
                W[(k_off+kk)*n + (i_off+ii)][col] += 1
                col += 1

    def add_block_schoolbook_2x2x1(i_off, j_off, k_off):
        # A: 2x2 (i_off, i_off+1) x (j_off, j_off+1)
        # B: 2x1 (j_off, j_off+1) x (k_off)
        # C: 2x1
        nonlocal col
        # Terms: A[i,j] * B[j,0]
        for ii in range(2):
            for jj in range(2):
                U[(i_off+ii)*m + (j_off+jj)][col] += 1
                V[(j_off+jj)*p + k_off][col] += 1
                W[k_off*n + (i_off+ii)][col] += 1
                col += 1

    def add_block_schoolbook_2x1x1(i_off, j_off, k_off):
        # A: 2x1 (i_off, i_off+1) x (j_off)
        # B: 1x1 (j_off) x (k_off)
        # C: 2x1
        nonlocal col
        # Terms: A[i,0] * B[0,0]
        for ii in range(2):
            U[(i_off+ii)*m + j_off][col] += 1
            V[j_off*p + k_off][col] += 1
            W[k_off*n + (i_off+ii)][col] += 1
            col += 1

    # Block 1: A1 B11 -> C1 (2x2, 2x2, 2x2) Strassen
    # i_off=0, j_off=0, k_off=0
    add_block_strassen(0, 0, 0)
    
    # Block 2: A2 B21 -> C1 (2x2, 2x2, 2x2) Strassen
    # A2: i in 0,1; j in 2,3. B21: j in 2,3; k in 0,1.
    # i_off=0, j_off=2, k_off=0
    add_block_strassen(0, 2, 0)
    
    # Block 3: A3 B31 -> C1 (2x1, 1x2, 2x2) Schoolbook
    # A3: i in 0,1; j in 4. B31: j in 4; k in 0,1.
    # i_off=0, j_off=4, k_off=0
    add_block_schoolbook_2x1x1x2(0, 4, 0)
    
    # Block 4: A1 B12 -> C2 (2x2, 2x2, 2x2) Strassen
    # A1: i in 0,1; j in 0,1. B12: j in 0,1; k in 2,3.
    # i_off=0, j_off=0, k_off=2
    add_block_strassen(0, 0, 2)
    
    # Block 5: A2 B22 -> C2 (2x2, 2x2, 2x2) Strassen
    # A2: i in 0,1; j in 2,3. B22: j in 2,3; k in 2,3.
    # i_off=0, j_off=2, k_off=2
    add_block_strassen(0, 2, 2)
    
    # Block 6: A3 B32 -> C2 (2x1, 1x2, 2x2) Schoolbook
    # A3: i in 0,1; j in 4. B32: j in 4; k in 2,3.
    # i_off=0, j_off=4, k_off=2
    add_block_schoolbook_2x1x1x2(0, 4, 2)
    
    # Block 7: A1 B13 -> C3 (2x2, 2x1, 2x1) Schoolbook
    # A1: i in 0,1; j in 0,1. B13: j in 0,1; k in 4.
    # i_off=0, j_off=0, k_off=4
    add_block_schoolbook_2x2x1(0, 0, 4)
    
    # Block 8: A2 B23 -> C3 (2x2, 2x1, 2x1) Schoolbook
    # A2: i in 0,1; j in 2,3. B23: j in 2,3; k in 4.
    # i_off=0, j_off=2, k_off=4
    add_block_schoolbook_2x2x1(0, 2, 4)
    
    # Block 9: A3 B33 -> C3 (2x1, 1x1, 2x1) Schoolbook
    # A3: i in 0,1; j in 4. B33: j in 4; k in 4.
    # i_off=0, j_off=4, k_off=4
    add_block_schoolbook_2x1x1(0, 4, 4)
    
    return {"U": U, "V": V, "W": W}
