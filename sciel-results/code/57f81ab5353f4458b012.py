import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 using 8 Strassen blocks
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = a11(b01 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # c00 = M1 - M4 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M6
    
    # Indices for A (U): 00=0, 01=1, 10=2, 11=3
    # Indices for B (V): 00=0, 01=1, 10=2, 11=3 (local to block)
    # Indices for C (W): 00=0, 01=1, 10=2, 11=3 (local to block)
    
    # U coefficients for M1..M7
    # M1: a00, a11 -> 1, 1
    # M2: a10, a11 -> 1, 1
    # M3: a00 -> 1
    # M4: a11 -> 1
    # M5: a00, a01 -> 1, 1
    # M6: a10, a00 -> 1, -1
    # M7: a01, a11 -> 1, -1
    
    U_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [1, 0, 1, 0], # M6 (a10 - a00) -> a00:-1, a10:1. Wait. M6 = (a10 - a00)(...). So a00:-1, a10:1.
        [0, 1, 0, -1] # M7 (a01 - a11) -> a01:1, a11:-1
    ]
    # Fix M6: a10 - a00. U[2]=1, U[0]=-1.
    U_coeffs[5] = [-1, 0, 1, 0]
    
    # V coefficients for M1..M7 (local block indices 0,1,2,3)
    # M1: b00, b11 -> 1, 1
    # M2: b00 -> 1
    # M3: b10, b11 -> 1, -1
    # M4: b01, b00 -> 1, -1
    # M5: b11 -> 1
    # M6: b00, b01 -> 1, 1
    # M7: b10, b11 -> 1, 1
    
    V_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1], # M3
        [0, 1, 0, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W coefficients for c00..c11
    # c00 = M1 - M4 + M7 -> M1:1, M4:-1, M7:1
    # c01 = M3 + M5 -> M3:1, M5:1
    # c10 = M2 + M4 -> M2:1, M4:1
    # c11 = M1 - M3 + M6 -> M1:1, M3:-1, M6:1
    
    W_coeffs = [
        [1, 0, 0, -1, 0, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, 0, -1, 0, 0, 1, 0]  # c11
    ]
    
    r = 0
    for l in range(8):
        # Block l uses B cols 2l, 2l+1 and C cols 2l, 2l+1
        # V local index 0 -> B[0, 2l] -> idx 0*p + 2l = 2l
        # V local index 1 -> B[0, 2l+1] -> idx 0*p + 2l+1 = 2l+1
        # V local index 2 -> B[1, 2l] -> idx 1*p + 2l = 16 + 2l
        # V local index 3 -> B[1, 2l+1] -> idx 1*p + 2l+1 = 16 + 2l+1
        
        # W local index 0 -> C[0, 2l] -> idx 2l*n + 0 = 2l*2 = 4l
        # W local index 1 -> C[0, 2l+1] -> idx (2l+1)*n + 0 = (2l+1)*2 = 4l+2
        # W local index 2 -> C[1, 2l] -> idx 2l*n + 1 = 4l+1
        # W local index 3 -> C[1, 2l+1] -> idx (2l+1)*n + 1 = 4l+3
        
        for s in range(7):
            # U
            for i in range(4):
                U[i, r] = U_coeffs[s][i]
            
            # V
            for j_local in range(4):
                v_idx = 0
                if j_local == 0: v_idx = 2*l
                elif j_local == 1: v_idx = 2*l + 1
                elif j_local == 2: v_idx = 16 + 2*l
                elif j_local == 3: v_idx = 16 + 2*l + 1
                V[v_idx, r] = V_coeffs[s][j_local]
            
            # W
            for k_local in range(4):
                w_idx = 0
                if k_local == 0: w_idx = 4*l
                elif k_local == 1: w_idx = 4*l + 2
                elif k_local == 2: w_idx = 4*l + 1
                elif k_local == 3: w_idx = 4*l + 3
                W[w_idx, r] = W_coeffs[k_local][s]
            
            r += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
