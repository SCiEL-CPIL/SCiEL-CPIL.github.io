import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2 matrices
    # U2, V2, W2 have shapes (4, 7), (4, 7), (4, 7)
    # Indices: U2[i*2+j, s], V2[j*2+k, s], W2[k*2+i, s]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # s=0: P0 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    # s=1: P1 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    # s=2: P2 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    # s=3: P3 = A11(B10-B01)
    U2[3][3] = 1
    V2[2][3] = 1; V2[1][3] = -1
    # s=4: P4 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    # s=5: P5 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    # s=6: P6 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    
    # W2[k*2+i, s]
    # C00 = P0 + P3 - P4 + P6
    W2[0][0] = 1; W2[0][3] = 1; W2[0][4] = -1; W2[0][6] = 1
    # C01 = P2 + P4
    W2[2][1] = 1; W2[2][4] = 1 # k=1, i=0 -> index 1*2+0=2
    # C10 = P1 + P3
    W2[0][3] = 1 # k=0, i=1 -> index 0*2+1=1. Wait.
    # Let's be precise. W2 index is k*2+i.
    # C10 (i=1, k=0) -> index 0*2+1 = 1.
    W2[1][1] = 1; W2[1][3] = 1
    # C11 (i=1, k=1) -> index 1*2+1 = 3.
    W2[3][0] = 1; W2[3][1] = -1; W2[3][2] = 1; W2[3][5] = 1
    
    # Global matrices
    rank = 27 * 7
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    col = 0
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                for s in range(7):
                    # Outer indices
                    # U row: i*6+j. i=2*i1+ii, j=2*j1+jj
                    # V row: j*6+k. j=2*j1+jj, k=2*k1+kk
                    # W row: k*6+i. k=2*k1+kk, i=2*i1+ii
                    
                    for ii in range(2):
                        for jj in range(2):
                            i = 2*i1 + ii
                            j = 2*j1 + jj
                            u_val = U2[ii*2+jj][s]
                            if u_val != 0:
                                U[i*6+j][col] = u_val
                            
                    for jj in range(2):
                        for kk in range(2):
                            j = 2*j1 + jj
                            k = 2*k1 + kk
                            v_val = V2[jj*2+kk][s]
                            if v_val != 0:
                                V[j*6+k][col] = v_val
                                
                    for kk in range(2):
                        for ii in range(2):
                            k = 2*k1 + kk
                            i = 2*i1 + ii
                            w_val = W2[kk*2+ii][s]
                            if w_val != 0:
                                W[k*6+i][col] = w_val
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
