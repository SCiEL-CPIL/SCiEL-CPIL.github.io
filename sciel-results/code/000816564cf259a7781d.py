import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    # Target rank 85
    # Block 1: 4x4x4 Strassen (rank 49)
    # Block 2: 4x4x1 Schoolbook (rank 16)
    # Block 3: 4x1x5 Schoolbook (rank 20)
    
    total_rank = 49 + 16 + 20
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # --- Generate 2x2x2 Strassen Decomposition ---
    # U_S, V_S, W_S shape (4, 7)
    # Indices i,j in 0..1 for 2x2
    # U_S[i*2+j, c]
    
    U_S = np.zeros((4, 7))
    V_S = np.zeros((4, 7))
    W_S = np.zeros((4, 7))
    
    # Strassen Formulas
    # M1 = (A11+A22)(B11+B22)
    # A11: i=0,j=0. A22: i=1,j=1.
    # B11: j=0,k=0. B22: j=1,k=1.
    # C11: i=0,k=0.
    
    # c=0: M1
    U_S[0*2+0, 0] += 1 # A11
    U_S[1*2+1, 0] += 1 # A22
    V_S[0*2+0, 0] += 1 # B11
    V_S[1*2+1, 0] += 1 # B22
    W_S[0*2+0, 0] += 1 # C11
    
    # c=1: M2 = (A21+A22)B11
    U_S[1*2+0, 1] += 1 # A21
    U_S[1*2+1, 1] += 1 # A22
    V_S[0*2+0, 1] += 1 # B11
    W_S[0*2+0, 1] += 1 # C11? No, M2 contributes to C21 and C22.
    # Wait, W_S[k*2+i, c].
    # M2 contributes to C21 (i=1, k=0) and C22 (i=1, k=1)?
    # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So M2 has +1 for C21 and -1 for C22.
    W_S[0*2+1, 1] += 1 # C21 (k=0, i=1)
    W_S[1*2+1, 1] += -1 # C22 (k=1, i=1)
    
    # c=2: M3 = A11(B21-B22)
    U_S[0*2+0, 2] += 1 # A11
    V_S[1*2+0, 2] += 1 # B21
    V_S[1*2+1, 2] += -1 # B22
    # M3 contributes to C12 (+1) and C22 (+1).
    W_S[0*2+1, 2] += 1 # C12 (k=0, i=1)? No. C12 is i=0, k=1.
    # W index is k*n+i. n=2.
    # C12: i=0, k=1. Index 1*2+0 = 2.
    # C22: i=1, k=1. Index 1*2+1 = 3.
    W_S[1*2+0, 2] += 1 # C12
    W_S[1*2+1, 2] += 1 # C22
    
    # c=3: M4 = A22(B12-B11)
    U_S[1*2+1, 3] += 1 # A22
    V_S[0*2+1, 3] += 1 # B12
    V_S[0*2+0, 3] += -1 # B11
    # M4 contributes to C11 (+1) and C21 (+1).
    # C11: i=0, k=0. Index 0.
    # C21: i=1, k=0. Index 2.
    W_S[0*2+0, 3] += 1 # C11
    W_S[0*2+1, 3] += 1 # C21
    
    # c=4: M5 = (A11+A12)B22
    U_S[0*2+0, 4] += 1 # A11
    U_S[0*2+1, 4] += 1 # A12
    V_S[1*2+1, 4] += 1 # B22
    # M5 contributes to C11 (-1) and C12 (+1).
    # C11: Index 0.
    # C12: Index 2.
    W_S[0*2+0, 4] += -1 # C11
    W_S[1*2+0, 4] += 1 # C12
    
    # c=5: M6 = (A21-A11)(B11+B12)
    U_S[1*2+0, 5] += 1 # A21
    U_S[0*2+0, 5] += -1 # A11
    V_S[0*2+0, 5] += 1 # B11
    V_S[0*2+1, 5] += 1 # B12
    # M6 contributes to C22 (+1).
    # C22: Index 3.
    W_S[1*2+1, 5] += 1 # C22
    
    # c=6: M7 = (A12-A22)(B21+B22)
    U_S[0*2+1, 6] += 1 # A12
    U_S[1*2+1, 6] += -1 # A22
    V_S[1*2+0, 6] += 1 # B21
    V_S[1*2+1, 6] += 1 # B22
    # M7 contributes to C11 (+1).
    # C11: Index 0.
    W_S[0*2+0, 6] += 1 # C11
    
    # --- Construct 4x4x4 Decomposition (Rank 49) ---
    # Kronecker product of U_S, V_S, W_S
    # U_4[i*4+j, c1*7+c2] = U_S[i*2+i2, c1] * U_S[j*2+j2, c2]
    # Wait, indices for 4x4 are i in 0..3, j in 0..3.
    # i = 2*i1 + i2.
    # U_4[i*4+j, c'] = U_S[(i//2)*2 + (j//2), c1] * U_S[(i%2)*2 + (j%2), c2]
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U_S[i1*2+j1, c1] * U_S[i2*2+j2, c2]
                            if val != 0:
                                U_4[i*4+j, c] = val
                            
                            val_v = V_S[i1*2+j1, c1] * V_S[i2*2+j2, c2]
                            if val_v != 0:
                                V_4[i*4+j, c] = val_v
                            
                            val_w = W_S[i1*2+j1, c1] * W_S[i2*2+j2, c2]
                            if val_w != 0:
                                W_4[i*4+j, c] = val_w
    
    # --- Fill U, V, W for Block 1 (4x4x4) ---
    # i in 0..3, j in 0..3, k in 0..3
    # U index: i*5 + j
    # V index: j*5 + k
    # W index: k*4 + i
    
    for c in range(49):
        for i in range(4):
            for j in range(4):
                u_val = U_4[i*4+j, c]
                if u_val != 0:
                    U[i*5+j, c] = u_val
                for k in range(4):
                    v_val = V_4[j*4+k, c]
                    if v_val != 0:
                        V[j*5+k, c] = v_val
                    w_val = W_4[k*4+i, c]
                    if w_val != 0:
                        W[k*4+i, c] = w_val
    
    # --- Fill U, V, W for Block 2 (4x4x1) ---
    # i in 0..3, j in 0..3, k=4
    # Schoolbook: 16 terms
    # c_offset = 49
    for i in range(4):
        for j in range(4):
            c = 49 + i*4 + j
            U[i*5+j, c] = 1
            V[j*5+4, c] = 1
            W[4*4+i, c] = 1
    
    # --- Fill U, V, W for Block 3 (4x1x5) ---
    # i in 0..3, j=4, k in 0..4
    # Schoolbook: 20 terms
    # c_offset = 49 + 16 = 65
    for i in range(4):
        for k in range(5):
            c = 65 + i*5 + k
            U[i*5+4, c] = 1
            V[4*5+k, c] = 1
            W[k*4+i, c] = 1
            
    # Convert to lists for return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
