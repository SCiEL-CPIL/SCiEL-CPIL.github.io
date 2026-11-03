import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank = 42
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 decomposition
    # U_strassen: (4, 7), V_strassen: (4, 7), W_strassen: (4, 7)
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1: (a00+a11)(b00+b11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1 # c00, c11
    
    # M2: (a10+a11)b00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[1, 1] = 1; W_s[3, 1] = -1 # c10, c11
    
    # M3: a00(b10-b11)
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[2, 2] = 1; W_s[3, 2] = 1 # c01, c11
    
    # M4: a11(b01-b00)
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[1, 3] = 1 # c00, c10
    
    # M5: (a00+a01)b11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[2, 4] = 1 # c00, c01
    
    # M6: (a10-a00)(b00+b10)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[2, 5] = 1
    W_s[3, 5] = 1 # c11
    
    # M7: (a01-a11)(b01+b11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[1, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1 # c00
    
    # We have 3 blocks of columns in B (size 4x2 each)
    # For each block, we split A into two 2x2 blocks (left, right)
    # A is 2x4. A_left is cols 0,1. A_right is cols 2,3.
    # B_block is 4x2. B_top is rows 0,1. B_bot is rows 2,3.
    
    col_offset = 0
    for block_idx in range(3):
        # Block corresponds to columns 2*block_idx and 2*block_idx+1 of B
        # and output columns 2*block_idx and 2*block_idx+1 of C
        
        # Sub-decomposition 1: A_left * B_top
        # A_left indices in U: i*4 + j, j in {0,1}
        # B_top indices in V: j*2 + k, j in {0,1}
        # C indices in W: k*2 + i, k in {0,1} (local) -> global k = 2*block_idx + k_local
        
        for r in range(7):
            col = col_offset + r
            # U
            for i in range(2):
                for j in range(2):
                    val = U_s[i*2+j, r]
                    if val != 0:
                        U[i*4 + j, col] = val
            # V
            for j in range(2):
                for k in range(2):
                    val = V_s[j*2+k, r]
                    if val != 0:
                        V[j*2 + k, col] = val
            # W
            for k in range(2):
                for i in range(2):
                    val = W_s[k*2+i, r]
                    if val != 0:
                        k_global = 2 * block_idx + k
                        W[k_global * 2 + i, col] = val

        # Sub-decomposition 2: A_right * B_bot
        # A_right indices in U: i*4 + j, j in {2,3} (local j' in {0,1})
        # B_bot indices in V: j*2 + k, j in {2,3} (local j' in {0,1})
        # C indices in W: same as above
        
        for r in range(7):
            col = col_offset + 7 + r
            # U
            for i in range(2):
                for j_local in range(2):
                    val = U_s[i*2+j_local, r]
                    if val != 0:
                        j_global = 2 + j_local
                        U[i*4 + j_global, col] = val
            # V
            for j_local in range(2):
                for k in range(2):
                    val = V_s[j_local*2+k, r]
                    if val != 0:
                        j_global = 2 + j_local
                        V[j_global*2 + k, col] = val
            # W
            for k in range(2):
                for i in range(2):
                    val = W_s[k*2+i, r]
                    if val != 0:
                        k_global = 2 * block_idx + k
                        W[k_global * 2 + i, col] = val

        col_offset += 14

    return {"U": U, "V": V, "W": W}
