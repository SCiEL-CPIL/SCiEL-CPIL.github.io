import numpy as np
import time
import random

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Attempts to find a low-rank decomposition for 6x6 matrix multiplication.
    Starts with the Kronecker product of Strassen (2x2, rank 7) and Schoolbook (3x3, rank 27),
    resulting in rank 189.
    Then attempts to reduce the rank via local search.
    """
    
    # --- Strassen 2x2 decomposition (rank 7) ---
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1
    
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[1,1] = 1; W2[3,1] = -1
    
    U2[0,2] = 1
    V2[1,2] = 1; V2[3,2] = -1
    W2[2,2] = 1; W2[3,2] = 1
    
    U2[3,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    W2[0,3] = 1; W2[1,3] = 1
    
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1; W2[2,4] = 1
    
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[3,5] = 1
    
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[0,6] = 1

    # --- Schoolbook 3x3 decomposition (rank 27) ---
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1

    # --- Build 6x6 decomposition via Kronecker product ---
    rank = 7 * 27
    U6 = np.zeros((36, rank))
    V6 = np.zeros((36, rank))
    W6 = np.zeros((36, rank))
    
    for c2 in range(7):
        for c3 in range(27):
            c = c2 * 27 + c3
            
            # U6
            for i2 in range(2):
                for j2 in range(2):
                    u2_val = U2[i2*2+j2, c2]
                    if u2_val == 0: continue
                    for i3 in range(3):
                        for j3 in range(3):
                            u3_val = U3[i3*3+j3, c3]
                            if u3_val == 0: continue
                            i = i2 * 3 + i3
                            j = j2 * 3 + j3
                            U6[i*6+j, c] = u2_val * u3_val
            
            # V6
            for j2 in range(2):
                for k2 in range(2):
                    v2_val = V2[j2*2+k2, c2]
                    if v2_val == 0: continue
                    for j3 in range(3):
                        for k3 in range(3):
                            v3_val = V3[j3*3+k3, c3]
                            if v3_val == 0: continue
                            j = j2 * 3 + j3
                            k = k2 * 3 + k3
                            V6[j*6+k, c] = v2_val * v3_val
            
            # W6
            for k2 in range(2):
                for i2 in range(2):
                    w2_val = W2[k2*2+i2, c2]
                    if w2_val == 0: continue
                    for k3 in range(3):
                        for i3 in range(3):
                            w3_val = W3[k3*3+i3, c3]
                            if w3_val == 0: continue
                            k = k2 * 3 + k3
                            i = i2 * 3 + i3
                            W6[k*6+i, c] = w2_val * w3_val

    # Convert to lists for return
    U_list = U6.tolist()
    V_list = V6.tolist()
    W_list = W6.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
