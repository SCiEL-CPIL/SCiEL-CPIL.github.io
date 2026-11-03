import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 45 decomposition
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Strassen part for 2x2 x 10 (5 blocks of 2x2x2)
    # Strassen coefficients for 2x2x2
    # M1..M7
    # U, V, W for each M
    # i, j, k are local 0..1
    # U_idx = i*2 + j
    # V_idx = j*10 + (2*b + k)
    # W_idx = (2*b + k)*2 + i
    
    strassen_blocks = 5
    col = 0
    for b in range(strassen_blocks):
        # 7 columns per block
        # M1: (A00+A11)(B00+B11)
        # U: i=0,j=0 -> 1; i=1,j=1 -> 1
        # V: j=0,k=0 -> 1; j=1,k=1 -> 1
        # W: k=0,i=0 -> 1; k=1,i=1 -> 1
        
        # We need to add contributions to U, V, W for each M
        # M1
        U[0*2+0][col] += 1
        U[1*2+1][col] += 1
        V[0*10 + (2*b+0)][col] += 1
        V[1*10 + (2*b+1)][col] += 1
        W[(2*b+0)*2 + 0][col] += 1
        W[(2*b+1)*2 + 1][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[1*2+0][col] += 1
        U[1*2+1][col] += 1
        V[0*10 + (2*b+0)][col] += 1
        W[(2*b+0)*2 + 1][col] += 1
        col += 1
        
        # M3: A00 B10
        U[0*2+0][col] += 1
        V[1*10 + (2*b+0)][col] += 1
        W[(2*b+0)*2 + 0][col] += 1
        col += 1
        
        # M4: A11 B11
        U[1*2+1][col] += 1
        V[1*10 + (2*b+1)][col] += 1
        W[(2*b+1)*2 + 1][col] += 1
        col += 1
        
        # M5: (A00+A01) B11
        U[0*2+0][col] += 1
        U[0*2+1][col] += 1
        V[1*10 + (2*b+1)][col] += 1
        W[(2*b+1)*2 + 0][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[1*2+0][col] += 1
        U[0*2+0][col] -= 1
        V[0*10 + (2*b+0)][col] += 1
        V[0*10 + (2*b+1)][col] += 1
        W[(2*b+0)*2 + 1][col] += 1
        W[(2*b+1)*2 + 1][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[0*2+1][col] += 1
        U[1*2+1][col] -= 1
        V[1*10 + (2*b+0)][col] += 1
        V[1*10 + (2*b+1)][col] += 1
        W[(2*b+0)*2 + 0][col] += 1
        W[(2*b+1)*2 + 0][col] += 1
        col += 1
        
    # 2. Rank-1 part for 2x1 x 10
    # i in 0..1, j=2, k in 0..9
    # U_idx = i*3 + 2
    # V_idx = 2*10 + k
    # W_idx = k*2 + i
    for k in range(10):
        for i in range(2):
            # Wait, rank 1 per k?
            # We need U[i, col], V[20+k, col], W[2k+i, col]
            # But U[i, col] is shared for i=0 and i=1?
            # No, for a fixed k, we need to satisfy:
            # sum_c U[i,c] V[20+k,c] W[2k+i,c] = 1
            # If we use one column per k, we can set:
            # U[0, col] = 1, U[1, col] = 1
            # V[20+k, col] = 1
            # W[2k+0, col] = 1, W[2k+1, col] = 1
            # Then for i=0: 1*1*1 = 1.
            # For i=1: 1*1*1 = 1.
            # So one column per k is sufficient.
            U[0*3+2][col] = 1
            U[1*3+2][col] = 1
            V[2*10+k][col] = 1
            W[k*2+0][col] = 1
            W[k*2+1][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
