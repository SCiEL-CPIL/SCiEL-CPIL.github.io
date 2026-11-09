import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 22
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (6, 22)
    # V: (m*p, rank) = (12, 22)
    # W: (p*n, rank) = (8, 22)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11-term decomposition for 2x3x2
    # U_11: 6x11, V_11: 6x11, W_11: 4x11
    
    U_11 = [[0] * 11 for _ in range(6)]
    V_11 = [[0] * 11 for _ in range(6)]
    W_11 = [[0] * 11 for _ in range(4)]
    
    # Terms 0-6: Strassen for j in {0,1}
    # r=0: M0
    U_11[0*2+0][0] = 1; U_11[1*2+1][0] = 1
    V_11[0*2+0][0] = 1; V_11[1*2+1][0] = 1
    W_11[0*2+0][0] = 1; W_11[1*2+1][0] = 1
    
    # r=1: M1
    U_11[1*2+0][1] = 1; U_11[1*2+1][1] = 1
    V_11[0*2+0][1] = 1
    W_11[1*2+0][1] = 1; W_11[1*2+1][1] = -1
    
    # r=2: M2
    U_11[0*2+0][2] = 1
    V_11[1*2+1][2] = 1; V_11[0*2+0][2] = -1
    W_11[0*2+1][2] = 1; W_11[1*2+1][2] = 1
    
    # r=3: M3
    U_11[1*2+1][3] = 1
    V_11[0*2+0][3] = 1; V_11[1*2+1][3] = -1
    W_11[0*2+0][3] = 1
    
    # r=4: M4
    U_11[0*2+0][4] = 1; U_11[0*2+1][4] = 1
    V_11[1*2+1][4] = 1
    W_11[0*2+0][4] = -1; W_11[0*2+1][4] = 1
    
    # r=5: M5
    U_11[1*2+0][5] = 1; U_11[0*2+0][5] = -1
    V_11[0*2+0][5] = 1; V_11[0*2+1][5] = 1
    W_11[1*2+1][5] = 1
    
    # r=6: M6
    U_11[0*2+1][6] = 1; U_11[1*2+1][6] = -1
    V_11[1*2+0][6] = 1; V_11[1*2+1][6] = 1
    W_11[0*2+0][6] = -1; W_11[1*2+0][6] = 1
    
    # Terms 7-10: Direct for j=2
    # r=7: i=0, k=0
    U_11[0*2+2][7] = 1
    V_11[2*2+0][7] = 1
    W_11[0*2+0][7] = 1
    
    # r=8: i=0, k=1
    U_11[0*2+2][8] = 1
    V_11[2*2+1][8] = 1
    W_11[0*2+1][8] = 1
    
    # r=9: i=1, k=0
    U_11[1*2+2][9] = 1
    V_11[2*2+0][9] = 1
    W_11[1*2+0][9] = 1
    
    # r=10: i=1, k=1
    U_11[1*2+2][10] = 1
    V_11[2*2+1][10] = 1
    W_11[1*2+1][10] = 1
    
    # Assemble 22-rank decomposition
    for c in range(11):
        # First block (k in 0,1)
        for i in range(n):
            for j in range(m):
                U[i*m+j][c] = U_11[i*2+j][c] if j < 2 else 0 # j=2 is handled in U_11 at index 2,3? 
                # Wait, U_11 index is i*2+j. j in 0,1,2.
                # My U_11 loop above used i*2+j where j was 0,1 for Strassen, and 2 for direct.
                # So U_11 is 6x11.
                # U[i*m+j] where m=3.
                # U_11 index: i*2+j.
                # Mapping: U[i*3+j][c] = U_11[i*2+j][c].
        
        for j in range(m):
            for k in range(2): # k in 0,1 for first block
                V[j*p+k][c] = V_11[j*2+k][c] if j < 2 else 0
                # V_11 index j*2+k. j in 0,1,2. k in 0,1.
                # Mapping: V[j*4+k][c] = V_11[j*2+k][c].
        
        for k in range(2):
            for i in range(n):
                W[k*n+i][c] = W_11[k*2+i][c]
                
        # Second block (k in 2,3) -> c+11
        c2 = c + 11
        for i in range(n):
            for j in range(m):
                U[i*m+j][c2] = U_11[i*2+j][c]
        
        for j in range(m):
            for k in range(2): # k_local in 0,1 -> k_global in 2,3
                V[j*p+(k+2)][c2] = V_11[j*2+k][c]
        
        for k in range(2):
            for i in range(n):
                W[(k+2)*n+i][c2] = W_11[k*2+i][c]

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
