"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9
    
    def strassen_2x2_tensor():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[2][3]=1; W[3][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[3][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    def strassen_4x4_tensor():
        U2, V2, W2 = strassen_2x2_tensor()
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        A_blocks = [[(0,0),(1,1)],[(1,0),(1,1)],[(0,0)],[(1,1)],[(0,0),(0,1)],[(1,0),(0,0)],[(0,1),(1,1)]]
        B_blocks = [[(0,0),(1,1)],[(0,0)],[(0,1),(1,1)],[(1,0),(0,0)],[(1,1)],[(0,0),(0,1)],[(1,0),(1,1)]]
        C_blocks = [[(0,0,1),(1,0,1),(1,1,-1),(0,1,1)],[(1,0,1),(1,1,-1)],[(0,1,1),(1,1,1)],[(0,0,1),(1,0,1)],[(0,0,-1),(0,1,1)],[(1,1,1)],[(0,0,1),(0,1,1)]]
        for t in range(7):
            for s in range(7):
                col = t*7+s
                for br,bc in A_blocks[t]:
                    for lr in range(2):
                        for lc in range(2):
                            li=lr*2+lc; gi=(br*2+lr)*4+(bc*2+lc)
                            c=U2[li][s]
                            if c: U[gi][col]+=c
                for br,bc in B_blocks[t]:
                    for lr in range(2):
                        for lc in range(2):
                            li=lr*2+lc; gi=(br*2+lr)*4+(bc*2+lc)
                            c=V2[li][s]
                            if c: V[gi][col]+=c
                for cbr,cbc,cc in C_blocks[t]:
                    for lr in range(2):
                        for lc in range(2):
                            li=lr*2+lc; gi=(cbr*2+lr)*4+(cbc*2+lc)
                            c=W2[li][s]*cc
                            if c: W[gi][col]+=c
        return U, V, W
    
    U4,V4,W4 = strassen_4x4_tensor()
    rank = 114
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    for i in range(16):
        for c in range(49):
            U[i][c]=U4[i][c]; U[i][49+c]=U4[i][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*9+k][c]=V4[j*4+k][c]
                V[j*9+4+k][49+c]=V4[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*9+i][c]=W4[k*4+i][c]
                W[(4+k)*9+i][49+c]=W4[k*4+i][c]
    for i in range(4):
        for j in range(4):
            c=98+i*4+j
            U[i*4+j][c]=1; V[j*9+8][c]=1; W[8*9+i][c]=1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
