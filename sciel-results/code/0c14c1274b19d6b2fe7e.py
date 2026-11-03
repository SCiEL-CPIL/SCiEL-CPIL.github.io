import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Generate Strassen 2x2
    def get_strassen_2x2():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    U2, V2, W2 = get_strassen_2x2()
    
    # Compose 4x4x4
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    a_combos = [
        [(0, 1), (3, 1)], [(2, 1), (3, 1)], [(0, 1)], [(3, 1)],
        [(0, 1), (1, 1)], [(2, 1), (0, -1)], [(1, 1), (3, -1)]
    ]
    b_combos = [
        [(0, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)], [(2, 1), (0, -1)],
        [(3, 1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]
    ]
    w_combos = [
        [(0, 1), (3, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)], [(0, 1), (2, 1)],
        [(0, -1), (1, 1)], [(3, 1)], [(0, 1)]
    ]
    
    for k in range(7):
        for c2 in range(7):
            c4 = k * 7 + c2
            
            for i2 in range(4):
                val = U2[i2, c2]
                if val == 0: continue
                i2_row = i2 // 2
                i2_col = i2 % 2
                for b_idx, coeff in a_combos[k]:
                    b_row = b_idx // 2
                    b_col = b_idx % 2
                    global_row = 2 * b_row + i2_row
                    global_col = 2 * b_col + i2_col
                    global_idx = global_row * 4 + global_col
                    U4[global_idx, c4] += val * coeff
            
            for i2 in range(4):
                val = V2[i2, c2]
                if val == 0: continue
                i2_row = i2 // 2
                i2_col = i2 % 2
                for b_idx, coeff in b_combos[k]:
                    b_row = b_idx // 2
                    b_col = b_idx % 2
                    global_row = 2 * b_row + i2_row
                    global_col = 2 * b_col + i2_col
                    global_idx = global_row * 4 + global_col
                    V4[global_idx, c4] += val * coeff
            
            for i2 in range(4):
                val = W2[i2, c2]
                if val == 0: continue
                i2_row = i2 // 2
                i2_col = i2 % 2
                for b_idx, coeff in w_combos[k]:
                    b_row = b_idx // 2
                    b_col = b_idx % 2
                    global_row = 2 * b_row + i2_row
                    global_col = 2 * b_col + i2_col
                    global_idx = global_row * 4 + global_col
                    W4[global_idx, c4] += val * coeff

    # Assemble Global U, V, W
    rank = 49 + 49 + 16
    U_global = np.zeros((16, rank))
    V_global = np.zeros((36, rank))
    W_global = np.zeros((36, rank))
    
    # Part 1: 0-48
    U_global[:, 0:49] = U4
    V_global[0:16, 0:49] = V4
    W_global[0:16, 0:49] = W4
    
    # Part 2: 49-97
    U_global[:, 49:98] = U4
    # V4 maps to rows 16-31 of V_global
    # V4 row r = j*4 + k. Global row = j*9 + (k+4)
    for r in range(16):
        j = r // 4
        k = r % 4
        global_r = j * 9 + (k + 4)
        V_global[global_r, 49:98] = V4[r, :]
    # W4 maps to rows 16-31 of W_global
    # W4 row r = k*4 + i. Global row = (k+4)*4 + i
    for r in range(16):
        k = r // 4
        i = r % 4
        global_r = (k + 4) * 4 + i
        W_global[global_r, 49:98] = W4[r, :]
        
    # Part 3: 98-113
    U3 = np.zeros((16, 16))
    V3 = np.zeros((4, 16))
    W3 = np.zeros((4, 16))
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U3[i * 4 + j, c] = 1
            V3[j, c] = 1
            W3[i, c] = 1
            
    U_global[:, 98:114] = U3
    # V3 maps to rows 32-35
    # V3 row r = j. Global row = j*9 + 8
    for r in range(4):
        j = r
        global_r = j * 9 + 8
        V_global[global_r, 98:114] = V3[r, :]
    # W3 maps to rows 32-35
    # W3 row r = i. Global row = 8*4 + i
    for r in range(4):
        i = r
        global_r = 8 * 4 + i
        W_global[global_r, 98:114] = W3[r, :]
        
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
