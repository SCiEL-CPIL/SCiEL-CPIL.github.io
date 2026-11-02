import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 42
    r = 42
    
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Strassen coefficients for 2x2x2
    # U_strassen[t][i][j]
    # V_strassen[t][j][k]
    # W_strassen[t][i][k]
    
    U_s = np.zeros((7, 2, 2))
    V_s = np.zeros((7, 2, 2))
    W_s = np.zeros((7, 2, 2))
    
    # t=0 (M1)
    U_s[0, 0, 0] = 1; U_s[0, 1, 1] = 1
    V_s[0, 0, 0] = 1; V_s[0, 1, 1] = 1
    W_s[0, 0, 0] = 1; W_s[0, 1, 1] = 1
    
    # t=1 (M2)
    U_s[1, 1, 0] = 1; U_s[1, 1, 1] = 1
    V_s[1, 0, 0] = 1
    W_s[1, 1, 0] = 1
    
    # t=2 (M3)
    U_s[2, 0, 0] = 1
    V_s[2, 0, 1] = 1; V_s[2, 1, 1] = -1
    W_s[2, 0, 1] = 1; W_s[2, 1, 1] = -1
    
    # t=3 (M4)
    U_s[3, 1, 1] = 1
    V_s[3, 1, 0] = 1; V_s[3, 0, 0] = -1
    W_s[3, 0, 0] = -1; W_s[3, 1, 0] = 1
    
    # t=4 (M5)
    U_s[4, 0, 0] = 1; U_s[4, 1, 0] = 1
    V_s[4, 1, 1] = 1
    W_s[4, 0, 1] = 1; W_s[4, 1, 1] = 1
    
    # t=5 (M6)
    U_s[5, 1, 1] = 1; U_s[5, 0, 0] = -1
    V_s[5, 0, 0] = 1; V_s[5, 1, 0] = 1
    W_s[5, 0, 0] = 1
    
    # t=6 (M7)
    U_s[6, 0, 1] = 1; U_s[6, 1, 1] = -1
    V_s[6, 1, 0] = 1; V_s[6, 1, 1] = 1
    W_s[6, 0, 0] = -1; W_s[6, 1, 1] = -1
    
    # Fill U, V, W
    col = 0
    for j_block in range(2):
        for k_block in range(3):
            for t in range(7):
                # U
                for i in range(2):
                    for j_loc in range(2):
                        j = j_block * 2 + j_loc
                        U[i*m + j, col] = U_s[t, i, j_loc]
                
                # V
                for j_loc in range(2):
                    for k_loc in range(2):
                        j = j_block * 2 + j_loc
                        k = k_block * 2 + k_loc
                        V[j*p + k, col] = V_s[t, j_loc, k_loc]
                
                # W
                for i in range(2):
                    for k_loc in range(2):
                        k = k_block * 2 + k_loc
                        W[k*n + i, col] = W_s[t, i, k_loc]
                        
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

