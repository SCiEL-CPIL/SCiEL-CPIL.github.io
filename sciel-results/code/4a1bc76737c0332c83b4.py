import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 8
    
    # Build Strassen 2x2x2 local factors
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # Standard Strassen for 2x2x2
    # Indices: U/V/W are 4x7. 
    # U rows: 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # V rows: 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    # W rows: 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Term 0: M0 = (A00+A11)(B00+B11)
    U_local[0,0] = 1; U_local[3,0] = 1
    V_local[0,0] = 1; V_local[3,0] = 1
    W_local[0,0] = 1; W_local[3,0] = 1
    
    # Term 1: M1 = (A10+A11)B00
    U_local[2,1] = 1; U_local[3,1] = 1
    V_local[0,1] = 1
    W_local[2,1] = 1; W_local[3,1] = -1
    
    # Term 2: M2 = A00(B10-B11)
    U_local[0,2] = 1
    V_local[1,2] = 1; V_local[3,2] = -1
    W_local[1,2] = 1; W_local[3,2] = 1
    
    # Term 3: M3 = A11(B01-B00)
    U_local[3,3] = 1
    V_local[2,3] = 1; V_local[0,3] = -1
    W_local[0,3] = 1; W_local[2,3] = 1
    
    # Term 4: M4 = (A00+A01)B11
    U_local[0,4] = 1; U_local[1,4] = 1
    V_local[3,4] = 1
    W_local[0,4] = -1; W_local[1,4] = 1
    
    # Term 5: M5 = (A10-A00)(B00+B01)
    U_local[2,5] = 1; U_local[0,5] = -1
    V_local[0,5] = 1; V_local[1,5] = 1
    W_local[3,5] = 1
    
    # Term 6: M6 = (A01-A11)(B10+B11)
    U_local[1,6] = 1; U_local[3,6] = -1
    V_local[2,6] = 1; V_local[3,6] = 1
    W_local[0,6] = 1
    
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    s = 0
                    for r in range(R):
                        s += U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
                    if s != 1:
                        return False
        return True

    total_rank = 56
    U = np.zeros((8, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((16, total_rank), dtype=int)
    
    for j_pair in range(2):  # j in {0,1} or {2,3}
        for k_pair in range(4):  # k-pair index
            base_r = (j_pair * 4 + k_pair) * 7
            
            # A mapping: for j_pair=0: A cols 0,1; j_pair=1: A cols 2,3
            if j_pair == 0:
                a_map = [0, 1, 4, 5]
            else:
                a_map = [2, 3, 6, 7]
            
            # B mapping: for j_pair=0: B rows 0,1; j_pair=1: B rows 2,3
            k0 = k_pair * 2
            if j_pair == 0:
                b_map = [0*8 + k0, 0*8 + k0+1, 1*8 + k0, 1*8 + k0+1]
            else:
                b_map = [2*8 + k0, 2*8 + k0+1, 3*8 + k0, 3*8 + k0+1]
            
            # C mapping: C[0,k0], C[0,k0+1], C[1,k0], C[1,k0+1]
            # W index: k*2 + i
            c_map = [k0*2+0, (k0+1)*2+0, k0*2+1, (k0+1)*2+1]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    if not verify(U, V, W):
        # Fallback to naive 64
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
