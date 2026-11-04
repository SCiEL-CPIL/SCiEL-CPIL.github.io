import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank 50
    # We will replace the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}) with Strassen rank 7
    
    # Indices for the 2x2x2 block
    i_block = [0, 1]
    j_block = [0, 1]
    k_block = [0, 1]
    
    # The 8 schoolbook terms in this block are:
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    # We remove these and add 7 Strassen terms.
    
    # Strassen decomposition for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the U, V, W matrices.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Let's build the full U, V, W with rank 49.
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen terms for the 2x2x2 block (indices 0,1 for i,j,k)
    # M1
    U[0*5+0][col] = 1; U[1*5+1][col] = 1
    V[0*5+0][col] = 1; V[1*5+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+1][col] = 1 # C11, C22
    # Wait, W index is k*n+i.
    # M1 contributes to C11 (k=0, i=0) and C22 (k=1, i=1).
    # So W[0*2+0] = 1, W[1*2+1] = 1.
    col += 1
    
    # M2
    U[1*5+1][col] = 1; U[1*5+0][col] = 1 # A21+A22? No, (A21+A22)B11.
    # A21 is i=1, j=0. A22 is i=1, j=1.
    U[1*5+0][col] = 1; U[1*5+1][col] = 1
    V[0*5+0][col] = 1 # B11
    W[0*2+1][col] = 1 # C21
    col += 1
    
    # M3
    U[0*5+0][col] = 1 # A11
    V[1*5+0][col] = 1; V[1*5+1][col] = -1 # B21 - B22
    W[0*2+1][col] = 1 # C12
    col += 1
    
    # M4
    U[1*5+1][col] = 1 # A22
    V[0*5+0][col] = 1; V[0*5+1][col] = -1 # B11 - B12
    W[0*2+0][col] = 1; W[1*2+1][col] = 1 # C11, C22
    col += 1
    
    # M5
    U[0*5+0][col] = 1; U[0*5+1][col] = 1 # A11+A12
    V[1*5+1][col] = 1 # B22
    W[0*2+0][col] = -1; W[1*2+1][col] = 1 # C11, C22
    col += 1
    
    # M6
    U[1*5+0][col] = 1; U[0*5+0][col] = -1 # A21 - A11
    V[0*5+0][col] = 1; V[0*5+1][col] = 1 # B11 + B12
    W[1*2+1][col] = 1 # C22
    col += 1
    
    # M7
    U[0*5+1][col] = 1; U[1*5+1][col] = -1 # A12 - A22
    V[1*5+0][col] = 1; V[1*5+1][col] = 1 # B21 + B22
    W[0*2+0][col] = 1; W[1*2+1][col] = 1 # C11, C22
    col += 1
    
    # Now add the remaining schoolbook terms
    # Exclude the 8 terms in the 2x2x2 block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in i_block and j in j_block and k in k_block:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
