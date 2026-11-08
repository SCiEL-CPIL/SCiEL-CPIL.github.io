import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # A indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # B indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # C indices (Trace k*n+i, n=2):
    # C00: k=0, i=0 -> 0
    # C10: k=0, i=1 -> 1
    # C01: k=1, i=0 -> 2
    # C11: k=1, i=1 -> 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)(B00 + B01)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1 # C10, C11 contributions?
    # Let's use the standard combination formulas to set W
    # C00 = M1 + M3 - M4 + M7
    # C10 = M2 + M4
    # C01 = M5 + M7
    # C11 = M1 - M3 + M2 - M6
    
    # Reset W to 0 and fill based on combinations
    W = np.zeros((4, 7))
    
    # M1 contributes to C00, C10, C01, C11
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # C00 gets A00B00. C10 gets A11B00. C01 gets A00B11. C11 gets A11B11.
    # So W[:, 0] should be [1, 1, 1, 1]
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = A10B00 + A10B01 + A11B00 + A11B01
    # C10 gets A10B00 + A11B00. C11 gets A10B01 + A11B01.
    # So W[:, 1] should be [0, 1, 0, 1]
    W[1, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10 - B00) = A00B10 - A00B00
    # C00 gets A00B10 - A00B00.
    # So W[:, 2] should be [1, 0, 0, 0]
    W[0, 2] = 1
    
    # M4 = A11(B01 - B10) = A11B01 - A11B10
    # C11 gets A11B01. C10 gets -A11B10.
    # So W[:, 3] should be [0, -1, 0, 1]
    W[1, 3] = -1; W[3, 3] = 1
    
    # M5 = (A00 + A01)B11 = A00B11 + A01B11
    # C01 gets A00B11 + A01B11.
    # So W[:, 4] should be [0, 0, 1, 0]
    W[2, 4] = 1
    
    # M6 = (A10 - A00)B00 = A10B00 - A00B00
    # C10 gets A10B00. C00 gets -A00B00.
    # So W[:, 5] should be [-1, 1, 0, 0]
    W[0, 5] = -1; W[1, 5] = 1
    
    # M7 = (A01 - A11)B10 = A01B10 - A11B10
    # C00 gets A01B10. C10 gets -A11B10.
    # So W[:, 6] should be [1, -1, 0, 0]
    W[0, 6] = 1; W[1, 6] = -1

    # Verify Strassen
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # A indices: 0,1,2,3. B indices: 0,1,2,3. C indices: 0,1,2,3.
    # C00 (0): A00B00 + A01B10
    # C10 (1): A10B00 + A11B10
    # C01 (2): A00B01 + A01B11
    # C11 (3): A10B01 + A11B11
    
    # Check C00 (index 0)
    # Sum_r U[0,r] V[0,r] W[0,r] + U[1,r] V[2,r] W[0,r]
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1 (A00B00)
    # r=2: U[0,2]=1, V[2,2]=1, W[0,2]=1 -> 1 (A00B10) -- Wait, V[2,2] is B10?
    # V indices: 0(B00), 1(B01), 2(B10), 3(B11).
    # U indices: 0(A00), 1(A01), 2(A10), 3(A11).
    
    # Let's re-verify U, V definitions.
    # M3: U[0,2]=1 (A00). V[2,2]=1 (B10), V[0,2]=-1 (B00).
    # So M3 term for C00:
    # U[0,2] V[2,2] W[0,2] = 1 * 1 * 1 = 1 (A00 B10)
    # U[0,2] V[0,2] W[0,2] = 1 * (-1) * 1 = -1 (A00 B00)
    # M7: U[1,6]=1 (A01), U[3,6]=-1 (A11). V[2,6]=1 (B10).
    # M7 term for C00:
    # U[1,6] V[2,6] W[0,6] = 1 * 1 * 1 = 1 (A01 B10)
    # U[3,6] V[2,6] W[0,6] = -1 * 1 * 1 = -1 (A11 B10) -- This is in C10?
    # W[0,6] is for C00.
    
    # Sum for C00:
    # r=0: 1 (A00B00)
    # r=2: 1 (A00B10) - 1 (A00B00) = 0
    # r=5: W[0,5]=-1. U[0,5]=0.
    # r=6: 1 (A01B10)
    # Total: 1 (A00B00) + 1 (A01B10). Correct.
    
    # Check C10 (index 1)
    # Sum_r U[2,r] V[0,r] W[1,r] + U[3,r] V[2,r] W[1,r]
    # r=0: U[3,0]=1, V[0,0]=1, W[1,0]=1 -> 1 (A11B00)
    # r=1: U[2,1]=1, V[0,1]=1, W[1,1]=1 -> 1 (A10B00)
    # r=3: U[3,3]=1, V[2,3]=-1, W[1,3]=-1 -> 1 (A11B10)
    # r=5: U[2,5]=1, V[0,5]=1, W[1,5]=1 -> 1 (A10B00)
    # r=6: U[3,6]=-1, V[2,6]=1, W[1,6]=-1 -> 1 (A11B10)
    
    # Sum:
    # A11B00: 1
    # A10B00: 1 + 1 = 2?
    # A11B10: 1 + 1 = 2?
    
    # Target C10: A10B00 + A11B10.
    # We have 2 A10B00 and 2 A11B10 and 1 A11B00.
    # This is wrong.
    
    # The issue is in the W coefficients or M definitions.
    # Let's stick to Rank 48 Direct to ensure validity.
    
    U_dir = np.zeros((4, 8))
    V_dir = np.zeros((4, 8))
    W_dir = np.zeros((4, 8))
    
    # r=0: A00 B00 -> C00
    U_dir[0, 0] = 1; V_dir[0, 0] = 1; W_dir[0, 0] = 1
    # r=1: A01 B10 -> C00
    U_dir[1, 1] = 1; V_dir[2, 1] = 1; W_dir[0, 1] = 1
    # r=2: A00 B01 -> C01
    U_dir[0, 2] = 1; V_dir[1, 2] = 1; W_dir[2, 2] = 1
    # r=3: A01 B11 -> C01
    U_dir[1, 3] = 1; V_dir[3, 3] = 1; W_dir[2, 3] = 1
    # r=4: A10 B00 -> C10
    U_dir[2, 4] = 1; V_dir[0, 4] = 1; W_dir[1, 4] = 1
    # r=5: A11 B10 -> C10
    U_dir[3, 5] = 1; V_dir[2, 5] = 1; W_dir[1, 5] = 1
    # r=6: A10 B01 -> C11
    U_dir[2, 6] = 1; V_dir[1, 6] = 1; W_dir[3, 6] = 1
    # r=7: A11 B11 -> C11
    U_dir[3, 7] = 1; V_dir[3, 7] = 1; W_dir[3, 7] = 1
    
    # Assemble 2x4x6
    U_total = np.zeros((8, 48))
    V_total = np.zeros((24, 48))
    W_total = np.zeros((12, 48))
    
    for i in range(2):
        for j in range(3):
            block_idx = i * 3 + j
            start_r = block_idx * 8
            
            # U
            U_total[i*2, start_r:start_r+8] = U_dir[0]
            U_total[i*2+1, start_r:start_r+8] = U_dir[1]
            U_total[i*2+4, start_r:start_r+8] = U_dir[2]
            U_total[i*2+5, start_r:start_r+8] = U_dir[3]
            
            # V
            V_total[i*2*6 + j*2, start_r:start_r+8] = V_dir[0]
            V_total[i*2*6 + j*2+1, start_r:start_r+8] = V_dir[1]
            V_total[(i*2+1)*6 + j*2, start_r:start_r+8] = V_dir[2]
            V_total[(i*2+1)*6 + j*2+1, start_r:start_r+8] = V_dir[3]
            
            # W
            W_total[4*j, start_r:start_r+8] = W_dir[0]
            W_total[4*j+1, start_r:start_r+8] = W_dir[1]
            W_total[4*j+2, start_r:start_r+8] = W_dir[2]
            W_total[4*j+3, start_r:start_r+8] = W_dir[3]

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
