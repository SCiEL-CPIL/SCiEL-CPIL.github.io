import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Helper to add a Strassen 2x2x2 block
    # i_offset: row offset in A (0 or 1? No, A is 2xN, so i is 0,1)
    # j_offset: col offset in A, row offset in B
    # k_offset: col offset in B, col offset in C
    # The block is A[i, j:j+2] * B[j:j+2, k:k+2] -> C[i, k:k+2]
    # Wait, A is 2xM. B is MxP.
    # Here M=5, P=5.
    # We are doing 2x4x4 part.
    # A is 2x4 (cols 0-3). B is 4x4 (rows 0-3, cols 0-3).
    # C is 2x4 (cols 0-3).
    
    def add_strassen_block(i_idx, j_off, k_off):
        nonlocal r
        # i_idx is the row in A (0 or 1)
        # j_off is the start col in A / start row in B
        # k_off is the start col in B / start col in C
        
        # Strassen 7 multiplications
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # m1
        U[i_idx * m + j_off][r] = 1
        U[i_idx * m + j_off + 1][r] = 1
        V[j_off * p + k_off][r] = 1
        V[j_off * p + k_off + 1][r] = 1
        W[k_off * n + i_idx][r] = 1
        W[(k_off + 1) * n + i_idx][r] = 1
        r += 1
        
        # m2
        U[i_idx * m + j_off + 1][r] = 1
        U[i_idx * m + j_off + 1][r] = 1 # Wait, a10 and a11?
        # a10 is row 1, col 0. a11 is row 1, col 1.
        # In A (2x2 block), rows are i_idx (only one row? No, A is 2xM).
        # Wait, my Strassen block is 2x2x2.
        # A is 2x2. Rows 0,1. Cols j_off, j_off+1.
        # So a00 = A[0, j_off], a01 = A[0, j_off+1]
        # a10 = A[1, j_off], a11 = A[1, j_off+1]
        
        # Correction:
        # m2 = (a10 + a11) * b00
        U[i_idx * m + j_off][r] = 0 # a10 is row 1?
        # The block is defined by i_idx?
        # No, the 2x2x2 block uses both rows of A.
        # So i_idx should not be fixed.
        # The function add_strassen_block should handle the 2 rows of A.
        
        # Let's redefine add_strassen_block to take j_off and k_off.
        # It uses A[0, j_off..j_off+1] and A[1, j_off..j_off+1].
        
        pass

    # Redefine
    def add_strassen(j_off, k_off):
        nonlocal r
        # A indices: 0*m+j_off, 0*m+j_off+1, 1*m+j_off, 1*m+j_off+1
        # B indices: j_off*p+k_off, j_off*p+k_off+1, (j_off+1)*p+k_off, (j_off+1)*p+k_off+1
        # C indices: k_off*n+0, k_off*n+1, (k_off+1)*n+0, (k_off+1)*n+1
        
        # m1 = (A[0,j]+A[1,j+1])? No.
        # a00=A[0,j], a01=A[0,j+1], a10=A[1,j], a11=A[1,j+1]
        
        # m1 = (a00+a11)(b00+b11)
        U[0*m + j_off][r] = 1
        U[1*m + j_off + 1][r] = 1
        V[j_off*p + k_off][r] = 1
        V[(j_off+1)*p + (k_off+1)][r] = 1
        W[k_off*n + 0][r] = 1
        W[(k_off+1)*n + 1][r] = 1
        r += 1
        
        # m2 = (a10+a11)b00
        U[1*m + j_off][r] = 1
        U[1*m + j_off + 1][r] = 1
        V[j_off*p + k_off][r] = 1
        W[k_off*n + 0][r] = 0 # m2 contributes to C10 and C11?
        # C10 = m2+m4. C11 = m1-m2+m3+m6.
        # So m2 contributes to C[1, k_off] and C[1, k_off+1]?
        # C[1, k_off] is index k_off*n + 1.
        # C[1, k_off+1] is index (k_off+1)*n + 1.
        W[k_off*n + 1][r] = 1
        W[(k_off+1)*n + 1][r] = -1
        r += 1
        
        # m3 = a00(b01-b11)
        U[0*m + j_off][r] = 1
        V[j_off*p + (k_off+1)][r] = 1
        V[(j_off+1)*p + (k_off+1)][r] = -1
        W[k_off*n + 0][r] = 1 # C01?
        # C01 = m3+m5.
        # C[0, k_off+1] is (k_off+1)*n + 0.
        W[(k_off+1)*n + 0][r] = 1
        r += 1
        
        # m4 = a11(b10-b00)
        U[1*m + j_off + 1][r] = 1
        V[(j_off+1)*p + k_off][r] = 1
        V[j_off*p + k_off][r] = -1
        W[k_off*n + 1][r] = 1 # C10
        W[(k_off+1)*n + 1][r] = 1 # C11?
        # C10 = m2+m4. C11 = m1-m2+m3+m6.
        # m4 contributes to C10 and C11?
        # Formula: C10 = m2+m4. C11 = ...
        # Wait, standard Strassen:
        # C00 = m1+m4-m5+m7
        # C01 = m3+m5
        # C10 = m2+m4
        # C11 = m1-m2+m3+m6
        # So m4 is in C00 and C10.
        W[k_off*n + 0][r] = 1 # C00
        W[k_off*n + 1][r] = 1 # C10
        r += 1
        
        # m5 = (a00+a01)b11
        U[0*m + j_off][r] = 1
        U[0*m + j_off + 1][r] = 1
        V[(j_off+1)*p + (k_off+1)][r] = 1
        W[k_off*n + 0][r] = -1 # C00
        W[(k_off+1)*n + 0][r] = 1 # C01
        r += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[1*m + j_off][r] = 1
        U[0*m + j_off][r] = -1
        V[j_off*p + k_off][r] = 1
        V[j_off*p + (k_off+1)][r] = 1
        W[(k_off+1)*n + 1][r] = 1 # C11
        r += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[0*m + j_off + 1][r] = 1
        U[1*m + j_off + 1][r] = -1
        V[(j_off+1)*p + k_off][r] = 1
        V[(j_off+1)*p + (k_off+1)][r] = 1
        W[k_off*n + 0][r] = 1 # C00
        r += 1

    # 4 Strassen blocks for 2x4x4
    add_strassen(0, 0) # j=0,1 k=0,1
    add_strassen(0, 2) # j=0,1 k=2,3
    add_strassen(2, 0) # j=2,3 k=0,1
    add_strassen(2, 2) # j=2,3 k=2,3
    
    # 2x4x1 (j=0..3, k=4)
    for i in range(2):
        for j in range(4):
            U[i*m + j][r] = 1
            V[j*p + 4][r] = 1
            W[4*n + i][r] = 1
            r += 1
            
    # 2x1x4 (j=4, k=0..3)
    for i in range(2):
        for k in range(4):
            U[i*m + 4][r] = 1
            V[4*p + k][r] = 1
            W[k*n + i][r] = 1
            r += 1
            
    # 2x1x1 (j=4, k=4)
    for i in range(2):
        U[i*m + 4][r] = 1
        V[4*p + 4][r] = 1
        W[4*n + i][r] = 1
        r += 1
        
    return {"U": U, "V": V, "W": W}
