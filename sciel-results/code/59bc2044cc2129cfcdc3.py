"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 24
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a simple product a[i,j] * b[j,k]
    def set_simple(i, j, k, col):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        
    # Strassen Core for A[0:2, 0:2] * B[0:2, 0:2]
    # Indices in U: i*m+j, V: j*p+k, W: k*n+i
    
    # M1 = (a00+a11)(b00+b11)
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*3+0][0] = 1; V[1*3+1][0] = 1
    W[0*3+0][0] = 1 # C00
    W[1*3+1][0] = 1 # C11
    
    # M2 = (a10+a11)b00
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*3+0][1] = 1
    W[0*3+1][1] = 1 # C10
    W[1*3+1][1] = -1 # C11
    
    # M3 = a00(b01-b11)
    U[0*3+0][2] = 1
    V[0*3+1][2] = 1; V[1*3+1][2] = -1
    W[0*3+0][2] = 1 # C01
    W[1*3+1][2] = 1 # C11
    
    # M4 = a11(b10-b00)
    U[1*3+1][3] = 1
    V[1*3+0][3] = 1; V[0*3+0][3] = -1
    W[0*3+0][3] = 1 # C00
    W[0*3+1][3] = 1 # C10
    
    # M5 = (a00+a01)b11
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*3+1][4] = 1
    W[0*3+0][4] = -1 # C00
    W[0*3+0][4] = -1 # Wait, W index for C00 is k*n+i = 0*3+0 = 0.
    # Correction: W[0,4] = -1
    W[1*3+0][4] = 1 # C01 -> k=1, i=0 -> 1*3+0 = 3. W[3,4]=1
    
    # M6 = (a10-a00)(b00+b01)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[0*3+0][5] = 1; V[0*3+1][5] = 1
    W[1*3+1][5] = 1 # C11
    
    # M7 = (a01-a11)(b10+b11)
    U[0*3+1][6] = 1; U[1*3+1][6] = -1
    V[1*3+0][6] = 1; V[1*3+1][6] = 1
    W[0*3+0][6] = 1 # C00
    
    # Border Products
    set_simple(0, 2, 0, 7)  # a02 b20
    set_simple(0, 2, 1, 8)  # a02 b21
    set_simple(1, 2, 0, 9)  # a12 b20
    set_simple(1, 2, 1, 10) # a12 b21
    set_simple(0, 0, 2, 11) # a00 b02
    set_simple(0, 1, 2, 12) # a01 b12
    set_simple(1, 0, 2, 13) # a10 b02
    set_simple(1, 1, 2, 14) # a11 b12
    set_simple(2, 0, 0, 15) # a20 b00
    set_simple(2, 1, 0, 16) # a21 b10
    set_simple(2, 0, 1, 17) # a20 b01
    set_simple(2, 1, 1, 18) # a21 b11
    set_simple(2, 0, 2, 19) # a20 b02
    set_simple(2, 1, 2, 20) # a21 b12
    set_simple(2, 2, 0, 21) # a22 b20
    set_simple(2, 2, 1, 22) # a22 b21
    set_simple(2, 2, 2, 23) # a22 b22
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
