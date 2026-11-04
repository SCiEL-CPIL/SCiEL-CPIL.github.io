import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Rank 55 decomposition
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A_L (2x2) * B_L (2x10) using Strassen blocks
    # A_L uses cols 0,1 of A. Indices j=0,1.
    # B_L uses rows 0,1 of B. Indices j'=0,1.
    # Split B_L into 5 blocks of size 2x2.
    for b in range(5):
        # B_L block b uses cols 2b, 2b+1 of B.
        # Strassen 7 terms
        # M1: (a00+a11)(b00+b11) -> C00+=, C11+=
        U[0*3+0, col] = 1; U[1*3+1, col] = 1
        V[0*10 + 2*b, col] = 1; V[1*10 + 2*b+1, col] = 1
        W[2*b*2 + 0, col] = 1; W[(2*b+1)*2 + 1, col] = 1
        col += 1
        
        # M2: (a10+a11)b00 -> C10+=, C11-=
        U[1*3+0, col] = 1; U[1*3+1, col] = 1
        V[0*10 + 2*b, col] = 1
        W[2*b*2 + 1, col] = 1; W[(2*b+1)*2 + 1, col] = -1
        col += 1
        
        # M3: a00(b01-b11) -> C01+=, C11+=
        U[0*3+0, col] = 1
        V[1*10 + 2*b+1, col] = 1; V[1*10 + 2*b+1, col] = -1 # Wait, b01 is row0 col1? 
        # Let's check Strassen definition carefully.
        # B = [[b00, b01], [b10, b11]]
        # M3 = a00 * (b01 - b11)
        # b01 is index (0,1) in block. Row 0, Col 1.
        # In full B, row 0, col 2b+1.
        # b11 is index (1,1) in block. Row 1, Col 1.
        # In full B, row 1, col 2b+1.
        V[0*10 + 2*b+1, col] = 1; V[1*10 + 2*b+1, col] = -1
        W[(2*b+1)*2 + 0, col] = 1; W[(2*b+1)*2 + 1, col] = 1
        col += 1
        
        # M4: a11(b10-b00) -> C00+=, C10+=
        U[1*3+1, col] = 1
        V[1*10 + 2*b, col] = 1; V[0*10 + 2*b, col] = -1
        W[2*b*2 + 0, col] = 1; W[2*b*2 + 1, col] = 1
        col += 1
        
        # M5: (a00+a10)b11 -> C00-=, C01+=
        U[0*3+0, col] = 1; U[1*3+0, col] = 1
        V[1*10 + 2*b+1, col] = 1
        W[2*b*2 + 0, col] = -1; W[(2*b+1)*2 + 0, col] = 1
        col += 1
        
        # M6: (a01-a00)(b00+b10) -> C11+=
        U[0*3+1, col] = 1; U[0*3+0, col] = -1
        V[0*10 + 2*b, col] = 1; V[1*10 + 2*b, col] = 1
        W[(2*b+1)*2 + 1, col] = 1
        col += 1
        
        # M7: (a11-a10)(b01+b11) -> C00+=
        U[1*3+1, col] = 1; U[1*3+0, col] = -1
        V[0*10 + 2*b+1, col] = 1; V[1*10 + 2*b+1, col] = 1
        W[2*b*2 + 0, col] = 1
        col += 1
        
    # Part 2: A_R (2x1) * B_R (1x10)
    # A_R uses col 2 of A. j=2.
    # B_R uses row 2 of B. j'=2.
    for i in range(n):
        for k in range(p):
            U[i*3+2, col] = 1
            V[2*10 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
            
    # col should be 55
    assert col == rank, f"Column count {col} != Rank {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
