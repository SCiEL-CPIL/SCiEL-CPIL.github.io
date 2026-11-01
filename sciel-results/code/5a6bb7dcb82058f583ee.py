import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen terms for 2x2x2 block
    # A indices: i in 0,1; j in j_offset, j_offset+1
    # B indices: j in j_offset, j_offset+1; k in k_offset, k_offset+1
    # C indices: i in 0,1; k in k_offset, k_offset+1
    def add_strassen(j_offset, k_offset):
        nonlocal col
        # P1 = (A00+A11)(B00+B11) -> C00+1, C11-1
        U[0][col] += 1; U[3][col] += 1
        V[j_offset*p + k_offset][col] += 1; V[(j_offset+1)*p + (k_offset+1)][col] += 1
        W[k_offset*n + 0][col] += 1; W[(k_offset+1)*n + 1][col] += -1
        col += 1
        
        # P2 = (A10+A11)B00 -> C10+1, C11-1
        U[1][col] += 1; U[3][col] += 1
        V[j_offset*p + k_offset][col] += 1
        W[k_offset*n + 1][col] += 1; W[(k_offset+1)*n + 1][col] += -1
        col += 1
        
        # P3 = A00(B01-B11) -> C01+1, C11+1
        U[0][col] += 1
        V[j_offset*p + (k_offset+1)][col] += 1; V[(j_offset+1)*p + (k_offset+1)][col] += -1
        W[(k_offset+1)*n + 0][col] += 1; W[(k_offset+1)*n + 1][col] += 1
        col += 1
        
        # P4 = A11(B10-B00) -> C00+1, C10+1
        U[3][col] += 1
        V[(j_offset+1)*p + k_offset][col] += 1; V[j_offset*p + k_offset][col] += -1
        W[k_offset*n + 0][col] += 1; W[k_offset*n + 1][col] += 1
        col += 1
        
        # P5 = (A00+A01)B11 -> C00-1, C01+1
        U[0][col] += 1; U[1][col] += 1
        V[(j_offset+1)*p + (k_offset+1)][col] += 1
        W[k_offset*n + 0][col] += -1; W[(k_offset+1)*n + 0][col] += 1
        col += 1
        
        # P6 = (A10-A00)(B00+B01) -> C11+1
        U[1][col] += 1; U[0][col] += -1
        V[j_offset*p + k_offset][col] += 1; V[j_offset*p + (k_offset+1)][col] += 1
        W[(k_offset+1)*n + 1][col] += 1
        col += 1
        
        # P7 = (A01-A11)(B10+B11) -> C00+1
        U[1][col] += 1; U[3][col] += -1
        V[(j_offset+1)*p + k_offset][col] += 1; V[(j_offset+1)*p + (k_offset+1)][col] += 1
        W[k_offset*n + 0][col] += 1
        col += 1

    # Part 1: A[:, 0:2] * B[0:2, 0:2]
    # j_offset=0, k_offset=0
    add_strassen(0, 0)
    
    # Part 2: A[:, 0:2] * B[0:2, 2:4]
    # j_offset=0, k_offset=2
    add_strassen(0, 2)
    
    # Part 3: A[:, 0:2] * B[0:2, 4:5]
    # Schoolbook 2x2x1
    # Terms:
    # A00 B04 -> C04
    # A10 B04 -> C14
    # A01 B14 -> C04
    # A11 B14 -> C14
    
    # A00 B04
    U[0][col] += 1
    V[0*p + 4][col] += 1
    W[4*n + 0][col] += 1
    col += 1
    
    # A10 B04
    U[3][col] += 1
    V[0*p + 4][col] += 1
    W[4*n + 1][col] += 1
    col += 1
    
    # A01 B14
    U[1][col] += 1
    V[1*p + 4][col] += 1
    W[4*n + 0][col] += 1
    col += 1
    
    # A11 B14
    U[4][col] += 1
    V[1*p + 4][col] += 1
    W[4*n + 1][col] += 1
    col += 1
    
    # Part 4: A[:, 2:3] * B[2:3, 0:5]
    # Schoolbook 2x1x5
    for i in range(2):
        for k in range(5):
            U[i*m + 2][col] += 1
            V[2*p + k][col] += 1
            W[k*n + i][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
