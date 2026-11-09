"""Matrix multiplication 2x3x4 rank-22 decomposition using Strassen for 2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    rank = 22
    
    # U: shape (n*m, rank) = (6, 22)
    # V: shape (m*p, rank) = (12, 22)  
    # W: shape (p*n, rank) = (8, 22)
    # 
    # A is 2x3: A[i,j] for i in 0..1, j in 0..2
    # B is 3x4: B[j,k] for j in 0..2, k in 0..3
    # C is 2x4: C[i,k] for i in 0..1, k in 0..3
    #
    # Partition: A = [A1 | A2], A1 is 2x2 (cols 0,1), A2 is 2x1 (col 2)
    #             B = [B1; B2], B1 is 2x4 (rows 0,1), B2 is 1x4 (row 2)
    # C = A1*B1 + A2*B2
    #
    # B1 = [B1a | B1b], B1a is 2x2 (cols 0,1), B1b is 2x2 (cols 2,3)
    # A1*B1a: Strassen 2x2x2 (7 terms)
    # A1*B1b: Strassen 2x2x2 (7 terms)
    # A2*B2: schoolbook 2x1x4 (8 terms)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen 2x2x2 decomposition helper
    # X is 2x2, Y is 2x2, Z = X*Y
    # X00=X[0,0], X01=X[0,1], X10=X[1,0], X11=X[1,1]
    # Y00=Y[0,0], Y01=Y[0,1], Y10=Y[1,0], Y11=Y[1,1]
    #
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y10-Y11)
    # M4 = X11*(Y01-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y10)
    # M7 = (X01-X11)*(Y01+Y11)
    #
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    #
    # In our tensor: X = A1 (2x2), Y = B1a or B1b (2x2)
    # A1[i, j] maps to U[i*m + j, r] where m=3, so A1 uses cols 0,1 of A
    #   A1[0,0] = A[0,0] -> U[0*3+0, r] = U[0, r]
    #   A1[0,1] = A[0,1] -> U[0*3+1, r] = U[1, r]
    #   A1[1,0] = A[1,0] -> U[1*3+0, r] = U[3, r]
    #   A1[1,1] = A[1,1] -> U[1*3+1, r] = U[4, r]
    #
    # B1a[j, k] for j in 0,1 and k in 0,1 maps to V[j*p + k, r] where p=4
    #   B1a[0,0] = B[0,0] -> V[0*4+0, r] = V[0, r]
    #   B1a[0,1] = B[0,1] -> V[0*4+1, r] = V[1, r]
    #   B1a[1,0] = B[1,0] -> V[1*4+0, r] = V[4, r]
    #   B1a[1,1] = B[1,1] -> V[1*4+1, r] = V[5, r]
    #
    # Output C[i, k] for i in 0,1 and k in 0,1 maps to W[k*n + i, r]
    #   C[0,0] -> W[0*2+0, r] = W[0, r]
    #   C[0,1] -> W[1*2+0, r] = W[2, r]
    #   C[1,0] -> W[0*2+1, r] = W[1, r]
    #   C[1,1] -> W[1*2+1, r] = W[3, r]
    
    def add_strassen(base_col, B_col_offset):
        """Add Strassen terms for A1 * B1[base_col:base_col+2]"""
        # B1a or B1b: B[j, base_col + k] for j in 0,1, k in 0,1
        # V indices: V[j*p + (base_col+k), r]
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,base_col+0]+B[1,base_col+1])
        # U: A[0,0]=U[0], A[1,1]=U[4] -> U[0]=1, U[4]=1
        # V: B[0,base_col]=V[base_col], B[1,base_col+1]=V[4+base_col+1]
        # W: contributes to C[0,base_col]=W[base_col*2+0], C[0,base_col+1]=W[(base_col+1)*2+0],
        #      C[1,base_col]=W[base_col*2+1], C[1,base_col+1]=W[(base_col+1)*2+1]
        # Z00=C[0,base_col], Z01=C[0,base_col+1], Z10=C[1,base_col], Z11=C[1,base_col+1]
        # M1: Z00+=M1, Z11+=M1 -> W[base_col*2+0]+=1, W[(base_col+1)*2+1]+=1
        
        col = base_col
        U[0][col] += 1   # A[0,0]
        U[4][col] += 1   # A[1,1]
        V[base_col][col] += 1       # B[0, base_col]
        V[4 + base_col + 1][col] += 1  # B[1, base_col+1]
        W[base_col * 2 + 0][col] += 1  # C[0, base_col]
        W[(base_col + 1) * 2 + 1][col] += 1  # C[1, base_col+1]
        
        # M2 = (A[1,0]+A[1,1]) * B[0, base_col]
        # U: A[1,0]=U[3], A[1,1]=U[4] -> U[3]=1, U[4]=1
        # V: B[0,base_col]=V[base_col]
        # W: Z10=C[1,base_col]=W[base_col*2+1] += M2
        col = base_col + 1
        U[3][col] += 1   # A[1,0]
        U[4][col] += 1   # A[1,1]
        V[base_col][col] += 1  # B[0, base_col]
        W[base_col * 2 + 1][col] += 1  # C[1, base_col]
        
        # M3 = A[0,0] * (B[1, base_col] - B[1, base_col+1])
        # U: A[0,0]=U[0]
        # V: B[1,base_col]=V[4+base_col], B[1,base_col+1]=V[4+base_col+1]
        # W: Z01=C[0,base_col+1]=W[(base_col+1)*2+0] += M3
        col = base_col + 2
        U[0][col] += 1   # A[0,0]
        V[4 + base_col][col] += 1       # B[1, base_col]
        V[4 + base_col + 1][col] -= 1   # B[1, base_col+1]
        W[(base_col + 1) * 2 + 0][col] += 1  # C[0, base_col+1]
        
        # M4 = A[1,1] * (B[0, base_col+1] - B[0, base_col])
        # U: A[1,1]=U[4]
        # V: B[0,base_col+1]=V[base_col+1], B[0,base_col]=V[base_col]
        # W: Z00=C[0,base_col]=W[base_col*2+0] += M4, Z10=C[1,base_col]=W[base_col*2+1] += M4
        col = base_col + 3
        U[4][col] += 1   # A[1,1]
        V[base_col + 1][col] += 1   # B[0, base_col+1]
        V[base_col][col] -= 1       # B[0, base_col]
        W[base_col * 2 + 0][col] += 1  # C[0, base_col]
        W[base_col * 2 + 1][col] += 1  # C[1, base_col]
        
        # M5 = (A[0,0]+A[0,1]) * B[1, base_col+1]
        # U: A[0,0]=U[0], A[0,1]=U[1]
        # V: B[1,base_col+1]=V[4+base_col+1]
        # W: Z00=C[0,base_col]=W[base_col*2+0] -= M5, Z01=C[0,base_col+1]=W[(base_col+1)*2+0] += M5
        col = base_col + 4
        U[0][col] += 1   # A[0,0]
        U[1][col] += 1   # A[0,1]
        V[4 + base_col + 1][col] += 1  # B[1, base_col+1]
        W[base_col * 2 + 0][col] -= 1  # C[0, base_col]
        W[(base_col + 1) * 2 + 0][col] += 1  # C[0, base_col+1]
        
        # M6 = (A[1,0]-A[0,0]) * (B[0, base_col] + B[1, base_col])
        # U: A[1,0]=U[3], A[0,0]=U[0]
        # V: B[0,base_col]=V[base_col], B[1,base_col]=V[4+base_col]
        # W: Z11=C[1,base_col+1]=W[(base_col+1)*2+1] += M6
        col = base_col + 5
        U[3][col] += 1   # A[1,0]
        U[0][col] -= 1   # A[0,0]
        V[base_col][col] += 1       # B[0, base_col]
        V[4 + base_col][col] += 1   # B[1, base_col]
        W[(base_col + 1) * 2 + 1][col] += 1  # C[1, base_col+1]
        
        # M7 = (A[0,1]-A[1,1]) * (B[0, base_col+1] + B[1, base_col+1])
        # U: A[0,1]=U[1], A[1,1]=U[4]
        # V: B[0,base_col+1]=V[base_col+1], B[1,base_col+1]=V[4+base_col+1]
        # W: Z00=C[0,base_col]=W[base_col*2+0] += M7, Z11=C[1,base_col+1]=W[(base_col+1)*2+1] += M7
        col = base_col + 6
        U[1][col] += 1   # A[0,1]
        U[4][col] -= 1   # A[1,1]
        V[base_col + 1][col] += 1   # B[0, base_col+1]
        V[4 + base_col + 1][col] += 1  # B[1, base_col+1]
        W[base_col * 2 + 0][col] += 1  # C[0, base_col]
        W[(base_col + 1) * 2 + 1][col] += 1  # C[1, base_col+1]
    
    # Strassen for A1 * B1a (B1a uses B cols 0,1)
    add_strassen(0, 0)
    
    # Strassen for A1 * B1b (B1b uses B cols 2,3)
    add_strassen(7, 2)
    
    # Schoolbook for A2 * B2: A2 is 2x1 (col 2 of A), B2 is 1x4 (row 2 of B)
    # C[i,k] += A[1,2] * B[2,k] for i in 0,1 and k in 0,3
    # U: A[1,2] = U[1*3+2] = U[5]
    # V: B[2,k] = V[2*4+k] = V[8+k]
    # W: C[i,k] = W[k*2+i]
    
    for i in range(2):
        for k in range(4):
            col = 14 + i * 4 + k
            U[5][col] += 1       # A[1,2]
            V[8 + k][col] += 1   # B[2,k]
            W[k * 2 + i][col] += 1  # C[i,k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
