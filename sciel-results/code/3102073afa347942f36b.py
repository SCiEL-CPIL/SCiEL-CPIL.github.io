import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # M1 = (p1+p4)(q1+q4)
    # M2 = (p3+p4)q1
    # M3 = p1(q2-q4)
    # M4 = p4(q3-q1)
    # M5 = (p1+p2)q4
    # M6 = (p3-p1)(q1+q2)
    # M7 = (p2-p4)(q3+q4)
    
    # Indices for A (2x2): 00, 01, 10, 11
    # Indices for B (2x2): 00, 01, 10, 11
    
    # We have 4 blocks of output (k=0,1; 2,3; 4,5; 6,7)
    # For each block, we have 14 products (7 for A1B1, 7 for A2B2)
    
    col = 0
    for block in range(4):
        # k_offset for B and W
        k_off = block * 2
        
        # A1 is A[:, 0:2], A2 is A[:, 2:4]
        # B1 is B[0:2, :], B2 is B[2:4, :]
        
        # Process A1 B1
        # A1 indices in A: 0,1 (i=0), 4,5 (i=1) -> wait, A is 2x4.
        # A[i, j]. i in 0,1. j in 0,1,2,3.
        # A1: j=0,1. A2: j=2,3.
        
        # B1 indices in B: j=0,1. B2: j=2,3.
        # B[j, k]. j in 0,1,2,3. k in 0,1,2,3,4,5,6,7.
        # For this block, k in k_off, k_off+1.
        
        # Strassen A1 B1
        # p1=A[0,0], p2=A[0,1], p3=A[1,0], p4=A[1,1]
        # q1=B[0,k_off], q2=B[0,k_off+1], q3=B[1,k_off], q4=B[1,k_off+1]
        
        # M1
        U[0][col] += 1; U[1][col] += 1 # p1+p4? No, p1 is A[0,0], p4 is A[1,1]
        # U index is i*m+j.
        # A[0,0] -> 0*4+0=0. A[1,1] -> 1*4+1=5.
        U[0][col] = 1
        U[5][col] = 1
        V[0*8 + k_off][col] = 1 # B[0, k_off]
        V[1*8 + k_off+1][col] = 1 # B[1, k_off+1]
        # W for M1 contributes to C11, C12, C21, C22
        # C11: +1, C12: 0, C21: 0, C22: +1
        W[(k_off)*2 + 0][col] = 1 # C[0, k_off]
        W[(k_off+1)*2 + 0][col] = 0
        W[(k_off)*2 + 1][col] = 0
        W[(k_off+1)*2 + 1][col] = 1 # C[1, k_off+1]
        # Wait, W index is k*n+i.
        # C[0, k_off] -> k=k_off, i=0 -> k_off*2+0
        # C[1, k_off+1] -> k=k_off+1, i=1 -> (k_off+1)*2+1
        col += 1
        
        # M2 = (p3+p4)q1
        U[4][col] = 1 # A[1,0]
        U[5][col] = 1 # A[1,1]
        V[0*8 + k_off][col] = 1
        W[(k_off)*2 + 1][col] = 1 # C21
        col += 1
        
        # M3 = p1(q2-q4)
        U[0][col] = 1
        V[0*8 + k_off+1][col] = 1
        V[1*8 + k_off+1][col] = -1
        W[(k_off+1)*2 + 0][col] = 1 # C12
        W[(k_off+1)*2 + 1][col] = 1 # C22
        col += 1
        
        # M4 = p4(q3-q1)
        U[5][col] = 1
        V[1*8 + k_off][col] = 1
        V[0*8 + k_off][col] = -1
        W[(k_off)*2 + 0][col] = 1 # C11
        W[(k_off)*2 + 1][col] = 1 # C21
        col += 1
        
        # M5 = (p1+p2)q4
        U[0][col] = 1
        U[1][col] = 1
        V[1*8 + k_off+1][col] = 1
        W[(k_off)*2 + 0][col] = -1 # C11
        W[(k_off+1)*2 + 1][col] = 1 # C22
        col += 1
        
        # M6 = (p3-p1)(q1+q2)
        U[4][col] = 1
        U[0][col] = -1
        V[0*8 + k_off][col] = 1
        V[0*8 + k_off+1][col] = 1
        W[(k_off+1)*2 + 1][col] = 1 # C22
        col += 1
        
        # M7 = (p2-p4)(q3+q4)
        U[1][col] = 1
        U[5][col] = -1
        V[1*8 + k_off][col] = 1
        V[1*8 + k_off+1][col] = 1
        W[(k_off)*2 + 0][col] = 1 # C11
        W[(k_off+1)*2 + 0][col] = 1 # C12
        col += 1
        
        # Process A2 B2
        # A2: j=2,3. A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
        # B2: j=2,3. B[2, k], B[3, k]
        
        # N1
        U[2][col] = 1
        U[7][col] = 1
        V[2*8 + k_off][col] = 1
        V[3*8 + k_off+1][col] = 1
        W[(k_off)*2 + 0][col] = 1
        W[(k_off+1)*2 + 1][col] = 1
        col += 1
        
        # N2
        U[6][col] = 1
        U[7][col] = 1
        V[2*8 + k_off][col] = 1
        W[(k_off)*2 + 1][col] = 1
        col += 1
        
        # N3
        U[2][col] = 1
        V[2*8 + k_off+1][col] = 1
        V[3*8 + k_off+1][col] = -1
        W[(k_off+1)*2 + 0][col] = 1
        W[(k_off+1)*2 + 1][col] = 1
        col += 1
        
        # N4
        U[7][col] = 1
        V[3*8 + k_off][col] = 1
        V[2*8 + k_off][col] = -1
        W[(k_off)*2 + 0][col] = 1
        W[(k_off)*2 + 1][col] = 1
        col += 1
        
        # N5
        U[2][col] = 1
        U[3][col] = 1
        V[3*8 + k_off+1][col] = 1
        W[(k_off)*2 + 0][col] = -1
        W[(k_off+1)*2 + 1][col] = 1
        col += 1
        
        # N6
        U[6][col] = 1
        U[2][col] = -1
        V[2*8 + k_off][col] = 1
        V[2*8 + k_off+1][col] = 1
        W[(k_off+1)*2 + 1][col] = 1
        col += 1
        
        # N7
        U[3][col] = 1
        U[7][col] = -1
        V[3*8 + k_off][col] = 1
        V[3*8 + k_off+1][col] = 1
        W[(k_off)*2 + 0][col] = 1
        W[(k_off+1)*2 + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
