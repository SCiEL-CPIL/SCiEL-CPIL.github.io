import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = 2, 3, 5
    # Block A: i in {0,1}, j in {0,1}, k in {0,1} - Strassen 2x2x2
    # Block B: i in {0,1}, j in {0,1}, k in {2,3} - Strassen 2x2x2
    # Block C: i in {0,1}, j in {0,1}, k in {4} - rank 1
    # Block D: i in {0,1}, j in {2}, k in {0,1,2,3,4} - rank 1
    rank = 7 + 7 + 1 + 1  # = 16
    
    U = [[0]*rank for _ in range(n*m)]  # 6 x 16
    V = [[0]*rank for _ in range(m*p)]  # 15 x 16
    W = [[0]*rank for _ in range(p*n)]  # 10 x 16
    
    # Helper: Strassen 2x2x2 decomposition
    # Local indices: a in {0,1} (row), b in {0,1} (col/row), c in {0,1} (col)
    # U local: [a*2+b, r], V local: [b*2+c, r], W local: [c*2+a, r]
    def strassen_offset(u_off, v_off, w_off, r_start):
        # P1
        U[u_off+0][r_start+0] = 1  # A[0,0]
        U[u_off+3][r_start+0] = 1  # A[1,1]
        V[v_off+0][r_start+0] = 1  # B[0,0]
        V[v_off+1][r_start+0] = 1  # B[0,1]
        W[w_off+0][r_start+0] = 1  # C[0,0]
        W[w_off+3][r_start+0] = 1  # C[1,1]
        
        # P2
        U[u_off+2][r_start+1] = 1  # A[1,0]
        U[u_off+3][r_start+1] = 1  # A[1,1]
        V[v_off+0][r_start+1] = 1  # B[0,0]
        W[w_off+2][r_start+1] = 1  # C[1,0]
        W[w_off+3][r_start+1] = -1  # C[1,1]
        
        # P3
        U[u_off+0][r_start+2] = 1  # A[0,0]
        V[v_off+1][r_start+2] = 1  # B[0,1]
        V[v_off+3][r_start+2] = -1  # B[1,1]
        W[w_off+1][r_start+2] = 1  # C[0,1]
        W[w_off+3][r_start+2] = 1  # C[1,1]
        
        # P4
        U[u_off+3][r_start+3] = 1  # A[1,1]
        V[v_off+2][r_start+3] = 1  # B[1,0]
        V[v_off+0][r_start+3] = -1  # B[0,0]
        W[w_off+0][r_start+3] = 1  # C[0,0]
        W[w_off+2][r_start+3] = 1  # C[1,0]
        
        # P5
        U[u_off+0][r_start+4] = 1  # A[0,0]
        U[u_off+1][r_start+4] = 1  # A[0,1]
        V[v_off+3][r_start+4] = 1  # B[1,1]
        W[w_off+0][r_start+4] = -1  # C[0,0]
        W[w_off+1][r_start+4] = 1  # C[0,1]
        
        # P6
        U[u_off+2][r_start+5] = 1  # A[1,0]
        U[u_off+0][r_start+5] = -1  # A[0,0]
        V[v_off+0][r_start+5] = 1  # B[0,0]
        V[v_off+1][r_start+5] = 1  # B[0,1]
        W[w_off+3][r_start+5] = 1  # C[1,1]
        
        # P7
        U[u_off+1][r_start+6] = 1  # A[0,1]
        U[u_off+3][r_start+6] = -1  # A[1,1]
        V[v_off+2][r_start+6] = 1  # B[1,0]
        V[v_off+3][r_start+6] = 1  # B[1,1]
        W[w_off+0][r_start+6] = 1  # C[0,0]
    
    # Block A: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*m+j = i*3+j, so rows 0,1,3,4
    # V index: j*p+k = j*5+k, so for j=0: k=0,1 → 0,1; j=1: k=0,1 → 5,6
    # W index: k*n+i = k*2+i, so for k=0: i=0,1 → 0,1; k=1: i=0,1 → 2,3
    
    # For Strassen, local (a,b,c) maps to:
    # U: a*3+b (a=0,1; b=0,1) → 0,1,3,4
    # V: b*5+c (b=0,1; c=0,1) → 0,1,5,6
    # W: c*2+a (c=0,1; a=0,1) → 0,1,2,3
    
    # I need to map local Strassen indices to global.
    # Let me just do it manually for each block.
    
    # Block A: r_start = 0
    # Local (a,b,c) → global: U[a*3+b], V[b*5+c], W[c*2+a]
    # P1: U[0,0]=U[0], U[1,1]=U[4]; V[0,0]=V[0], V[0,1]=V[1]; W[0,0]=W[0], W[1,1]=W[3]
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2: U[1,0]=U[3], U[1,1]=U[4]; V[0,0]=V[0]; W[1,0]=W[2], W[1,1]=W[3]
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # P3: U[0,0]=U[0]; V[0,1]=V[1], V[1,1]=V[6]; W[0,1]=W[1], W[1,1]=W[3]
    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # P4: U[1,1]=U[4]; V[1,0]=V[5], V[0,0]=V[0]; W[0,0]=W[0], W[1,0]=W[2]
    U[4][3] = 1
    V[5][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # P5: U[0,0]=U[0], U[0,1]=U[1]; V[1,1]=V[6]; W[0,0]=W[0], W[0,1]=W[1]
    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # P6: U[1,0]=U[3], U[0,0]=U[0]; V[0,0]=V[0], V[0,1]=V[1]; W[1,1]=W[3]
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7: U[0,1]=U[1], U[1,1]=U[4]; V[1,0]=V[5], V[1,1]=V[6]; W[0,0]=W[0]
    U[1][6] = 1; U[4][6] = -1
    V[5][6] = 1; V[6][6] = 1
    W[0][6] = 1
    
    # Block B: i in {0,1}, j in {0,1}, k in {2,3}
    # r_start = 7
    # U: a*3+b → 0,1,3,4 (same as block A)
    # V: b*5+c where c=2,3 → b=0: 2,3; b=1: 7,8
    # W: c*2+a where c=2,3 → c=2: 4,5; c=3: 6,7
    
    U[0][7] = 1; U[4][7] = 1
    V[2][7] = 1; V[3][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[3][8] = 1; U[4][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[8][9] = -1
    W[5][9] = 1; W[7][9] = 1
    
    U[4][10] = 1
    V[7][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[8][11] = 1
    W[4][11] = -1; W[5][11] = 1
    
    U[3][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[4][13] = -1
    V[7][13] = 1; V[8][13] = 1
    W[4][13] = 1
    
    # Block C: i in {0,1}, j in {0,1}, k in {4}
    # r = 14
    # U[i*3+j, 14] = 1 for all i,j in {0,1} → U[0,1,3,4][14] = 1
    # V[j*5+4, 14] = 1 for j in {0,1} → V[4,9][14] = 1
    # W[4*2+i, 14] = 1 for i in {0,1} → W[8,9][14] = 1
    U[0][14] = 1; U[1][14] = 1; U[3][14] = 1; U[4][14] = 1
    V[4][14] = 1; V[9][14] = 1
    W[8][14] = 1; W[9][14] = 1
    
    # Block D: i in {0,1}, j in {2}, k in {0,1,2,3,4}
    # r = 15
    # U[i*3+2, 15] = 1 for i in {0,1} → U[2,5][15] = 1
    # V[2*5+k, 15] = 1 for k in {0,1,2,3,4} → V[10,11,12,13,14][15] = 1
    # W[k*2+i, 15] = 1 for all k,i → W[0..9][15] = 1
    U[2][15] = 1; U[5][15] = 1
    for k in range(5):
        V[10+k][15] = 1
    for k in range(5):
        for i in range(2):
            W[k*2+i][15] = 1
    
    return {"U": U, "V": V, "W": W}
