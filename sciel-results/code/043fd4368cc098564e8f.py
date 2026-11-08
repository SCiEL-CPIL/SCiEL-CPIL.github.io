import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate 49 terms for 4x4x4
    u_2x2 = [
        [1,0,1,0],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,-1],
        [0,1,0,-1]
    ]
    v_2x2 = [
        [1,1,1,1],
        [1,0,0,0],
        [0,1,0,-1],
        [0,0,1,-1],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1]
    ]
    w_2x2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    c_X = [None]*7
    c_Y = [None]*7
    for k in range(7):
        c_X[k] = [np.zeros((2,2)) for _ in range(4)]
        c_Y[k] = [np.zeros((2,2)) for _ in range(4)]
        
    c_X[0][0] = np.eye(2); c_X[0][3] = np.eye(2)
    c_Y[0][0] = np.eye(2); c_Y[0][3] = np.eye(2)
    
    c_X[1][2] = np.eye(2); c_X[1][3] = np.eye(2)
    c_Y[1][0] = np.eye(2)
    
    c_X[2][0] = np.eye(2)
    c_Y[2][1] = np.eye(2); c_Y[2][3] = -np.eye(2)
    
    c_X[3][3] = np.eye(2)
    c_Y[3][2] = np.eye(2); c_Y[3][0] = -np.eye(2)
    
    c_X[4][0] = np.eye(2); c_X[4][1] = np.eye(2)
    c_Y[4][3] = np.eye(2)
    
    c_X[5][2] = np.eye(2); c_X[5][0] = -np.eye(2)
    c_Y[5][0] = np.eye(2); c_Y[5][1] = np.eye(2)
    
    c_X[6][1] = np.eye(2); c_X[6][3] = -np.eye(2)
    c_Y[6][2] = np.eye(2); c_Y[6][3] = np.eye(2)
    
    w_44 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    for k in range(7):
        for j in range(7):
            r = k*7 + j
            for b in range(4):
                for e in range(4):
                    U_44[b*4 + e, r] = u_2x2[j][e] * c_X[k][b][e//2, e%2]
                    V_44[b*4 + e, r] = v_2x2[j][e] * c_Y[k][b][e//2, e%2]
            for c in range(4):
                for d in range(4):
                    W_44[c*4 + d, r] = w_44[c][k] * w_2x2[d][j]
                    
    # 2. Construct 4x4x9 decomposition
    rank = 49 + 49 + 16
    U_total = np.zeros((16, rank))
    V_total = np.zeros((36, rank))
    W_total = np.zeros((36, rank))
    
    # Part 1: First 4 columns
    for r in range(49):
        U_total[:, r] = U_44[:, r]
        for j in range(4):
            for k in range(4):
                V_total[j*9 + k, r] = V_44[j*4 + k, r]
        for k in range(4):
            for i in range(4):
                W_total[k*4 + i, r] = W_44[k*4 + i, r]
                
    # Part 2: Next 4 columns
    for r in range(49):
        r2 = 49 + r
        U_total[:, r2] = U_44[:, r]
        for j in range(4):
            for k in range(4):
                V_total[j*9 + (4+k), r2] = V_44[j*4 + k, r]
        for k in range(4):
            for i in range(4):
                W_total[(4+k)*4 + i, r2] = W_44[k*4 + i, r]
                
    # Part 3: Last 1 column
    for i in range(4):
        for j in range(4):
            r = 98 + i*4 + j
            U_total[i*4 + j, r] = 1
            V_total[j*9 + 8, r] = 1
            W_total[8*4 + i, r] = 1
            
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
