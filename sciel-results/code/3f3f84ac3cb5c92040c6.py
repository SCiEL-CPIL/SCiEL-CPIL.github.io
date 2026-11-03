import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen 2x2x2 decomposition
    # U, V, W are 4x7 matrices
    # Indices: i,j in {0,1} for U; j,k in {0,1} for V; k,i in {0,1} for W
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # M3: A00(B01-B11) -> C01
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # M4: A11(B10-B00) -> C10
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 2. Construct 4x4x4 decomposition via tensor product
    # Rank 49
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            u_idx = i * 4 + j
                            u_val = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
                            U4[u_idx, c] = u_val
                            
                            v_idx = j * 4 + (k1 * 2 + k2) # Wait, V depends on j,k
                            # V row index in 4x4 is j*4 + k
                            # V2 row index is j*2 + k
                            pass # Need to loop k too

    # Let's do it properly with loops
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    
                                    # U
                                    u_idx = i * 4 + j
                                    u_val = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
                                    U4[u_idx, c] = u_val
                                    
                                    # V
                                    v_idx = j * 4 + k
                                    v_val = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
                                    V4[v_idx, c] = v_val
                                    
                                    # W
                                    w_idx = k * 4 + i
                                    w_val = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]
                                    W4[w_idx, c] = w_val

    # 3. Construct 4x5x5 matrices
    rank = 52
    U = np.zeros((20, rank), dtype=int)
    V = np.zeros((25, rank), dtype=int)
    W = np.zeros((20, rank), dtype=int)
    
    # Copy 4x4x4 part
    # U rows: i*4+j -> i*5+j
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, 0:49] = U4[i * 4 + j, :]
            
    # V rows: j*4+k -> j*5+k
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, 0:49] = V4[j * 4 + k, :]
            
    # W rows: k*4+i -> k*4+i (since n=4 in both)
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, :]
            
    # Add extra terms
    # Term 1: j=4, k in 0..3 (Rank 1)
    for i in range(4):
        U[i * 5 + 4, 49] = 1
    for k in range(4):
        V[4 * 5 + k, 49] = 1
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 49] = 1
            
    # Term 2: j in 0..3, k=4 (Rank 1)
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, 50] = 1
    for j in range(4):
        V[j * 5 + 4, 50] = 1
    for i in range(4):
        W[4 * 4 + i, 50] = 1
        
    # Term 3: j=4, k=4 (Rank 1)
    for i in range(4):
        U[i * 5 + 4, 51] = 1
    V[4 * 5 + 4, 51] = 1
    for i in range(4):
        W[4 * 4 + i, 51] = 1

    # Convert to lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
