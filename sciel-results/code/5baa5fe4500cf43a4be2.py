"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_factors_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 matrix multiplication with rank 7 (Strassen).
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    # Strassen constants
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11) b00
    # M3 = a00 (b10 - b11)
    # M4 = a11 (b01 - b00)
    # M5 = (a00 + a01) b11
    # M6 = (a10 - a00)(b00 + b10)
    # M7 = (a01 - a11)(b01 + b11)
    
    # U factors (for A)
    # U is 4x7. Rows indexed by (i,j) flattened.
    # i in {0,1}, j in {0,1}
    # c in {0..6}
    
    U = np.zeros((4, 7))
    # M1: a00, a11 -> U[0,0]=1, U[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    # M2: a10, a11 -> U[2,1]=1, U[3,1]=1
    U[2, 1] = 1; U[3, 1] = 1
    # M3: a00 -> U[0,2]=1
    U[0, 2] = 1
    # M4: a11 -> U[3,3]=1
    U[3, 3] = 1
    # M5: a00, a01 -> U[0,4]=1, U[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    # M6: a10, a00 -> U[2,5]=1, U[0,5]=-1
    U[2, 5] = 1; U[0, 5] = -1
    # M7: a01, a11 -> U[1,6]=1, U[3,6]=-1
    U[1, 6] = 1; U[3, 6] = -1
    
    # V factors (for B)
    # V is 4x7. Rows indexed by (j,k) flattened.
    # j in {0,1}, k in {0,1}
    
    V = np.zeros((4, 7))
    # M1: b00, b11 -> V[0,0]=1, V[3,0]=1
    V[0, 0] = 1; V[3, 0] = 1
    # M2: b00 -> V[0,1]=1
    V[0, 1] = 1
    # M3: b10, b11 -> V[2,2]=1, V[3,2]=-1
    V[2, 2] = 1; V[3, 2] = -1
    # M4: b01, b00 -> V[1,3]=1, V[0,3]=-1
    V[1, 3] = 1; V[0, 3] = -1
    # M5: b11 -> V[3,4]=1
    V[3, 4] = 1
    # M6: b00, b10 -> V[0,5]=1, V[2,5]=1
    V[0, 5] = 1; V[2, 5] = 1
    # M7: b01, b11 -> V[1,6]=1, V[3,6]=1
    V[1, 6] = 1; V[3, 6] = 1
    
    # W factors (for C)
    # W is 4x7. Rows indexed by (k,i) flattened.
    # k in {0,1}, i in {0,1}
    
    W = np.zeros((4, 7))
    # C00 = M1 - M4 + M5 - M6 + M7
    W[0, 0] = 1; W[0, 3] = -1; W[0, 4] = 1; W[0, 5] = -1; W[0, 6] = 1
    # C01 = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    # C10 = M3 + M6
    W[2, 2] = 1; W[2, 5] = 1
    # C11 = M1 - M3 + M5 - M7
    W[3, 0] = 1; W[3, 2] = -1; W[3, 4] = 1; W[3, 6] = -1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # 1. Get Strassen factors for 2x2x2
    U_s, V_s, W_s = _strassen_factors_2x2x2()
    
    # 2. Construct 2x2x6 factors
    # Rank is 3 * 7 = 21
    R_226 = 21
    U_226 = np.zeros((4, R_226))
    V_226 = np.zeros((12, R_226)) # m=2, p=6 -> 2*6=12
    W_226 = np.zeros((12, R_226)) # p=6, n=2 -> 6*2=12
    
    # U is same for all 3 blocks of B (cols 0-1, 2-3, 4-5)
    for block in range(3):
        start = block * 7
        U_226[:, start:start+7] = U_s
        
    # V: rows (j,k). j in {0,1}, k in {0..5}
    # For block b, k in {2b, 2b+1}
    # V[j, 2b + k_mod] = V_s[j, k_mod]
    for block in range(3):
        start = block * 7
        for j in range(2):
            for k_mod in range(2):
                k_global = 2 * block + k_mod
                V_226[j * 6 + k_global, start:start+7] = V_s[j, k_mod]
                
    # W: rows (k,i). k in {0..5}, i in {0,1}
    # For block b, k in {2b, 2b+1}
    # W[2b + k_mod, i] = W_s[k_mod, i]
    for block in range(3):
        start = block * 7
        for k_mod in range(2):
            k_global = 2 * block + k_mod
            for i in range(2):
                W_226[k_global * 2 + i, start:start+7] = W_s[k_mod, i]

    # 3. Construct 2x4x6 factors
    # Rank is 2 * 21 = 42
    R_total = 42
    U_total = np.zeros((8, R_total)) # n=2, m=4 -> 8
    V_total = np.zeros((24, R_total)) # m=4, p=6 -> 24
    W_total = np.zeros((12, R_total)) # p=6, n=2 -> 12
    
    # Split A (2x4) into A_L (2x2) and A_R (2x2)
    # Split B (4x6) into B_T (2x6) and B_B (2x6)
    # C = A_L B_T + A_R B_B
    
    # Part 1: A_L B_T (uses j in {0,1})
    # U_total[i, j] for j in {0,1} comes from U_226[i, j]
    for i in range(2):
        for j in range(2):
            U_total[i * 4 + j, 0:21] = U_226[i, j]
            
    # V_total[j, k] for j in {0,1} comes from V_226[j, k]
    for j in range(2):
        for k in range(6):
            V_total[j * 6 + k, 0:21] = V_226[j, k]
            
    # W_total[k, i] comes from W_226[k, i]
    for k in range(6):
        for i in range(2):
            W_total[k * 2 + i, 0:21] = W_226[k, i]
            
    # Part 2: A_R B_B (uses j in {2,3})
    # U_total[i, j] for j in {2,3} comes from U_226[i, j-2]
    for i in range(2):
        for j in range(2, 4):
            U_total[i * 4 + j, 21:42] = U_226[i, j-2]
            
    # V_total[j, k] for j in {2,3} comes from V_226[j-2, k]
    for j in range(2, 4):
        for k in range(6):
            V_total[j * 6 + k, 21:42] = V_226[j-2, k]
            
    # W_total[k, i] comes from W_226[k, i]
    for k in range(6):
        for i in range(2):
            W_total[k * 2 + i, 21:42] = W_226[k, i]

    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

# EVOLVE-BLOCK-END
