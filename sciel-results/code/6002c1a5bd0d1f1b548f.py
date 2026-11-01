import numpy as np

def construct():
    n, m, p = 2, 2, 9
    rank = 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2x2 decomposition
    # U indices: 0:a, 1:b, 2:c, 3:d
    # V indices: 0:e, 1:f, 2:g, 3:h
    # W indices: 0:p, 1:q, 2:r, 3:s
    strassen_U = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 1, 0, 0], # m5
        [-1, 0, 1, 0], # m6
        [0, 1, 0, -1]  # m7
    ]
    strassen_V = [
        [1, 1, 0, 0], # m1
        [1, 0, 0, 0], # m2
        [0, 1, 0, -1], # m3
        [-1, 0, 1, 0], # m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    strassen_W = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 0], # m2
        [0, 1, 0, 1], # m3
        [1, 0, 1, 0], # m4
        [-1, 1, 0, 0], # m5
        [0, 0, 0, 1], # m6
        [1, 0, 0, 0]  # m7
    ]
    
    # Schoolbook 2x2x1 decomposition
    # U indices: 0:a, 1:b, 2:c, 3:d
    # V indices: 0:e, 1:g
    # W indices: 0:C0, 1:C1
    school_U = [
        [1, 0, 0, 0], # ae
        [0, 1, 0, 0], # bg
        [0, 0, 1, 0], # ce
        [0, 0, 0, 1]  # dg
    ]
    school_V = [
        [1, 0], # ae
        [0, 1], # bg
        [1, 0], # ce
        [0, 1]  # dg
    ]
    school_W = [
        [1, 0], # ae -> C0
        [1, 0], # bg -> C0
        [0, 1], # ce -> C1
        [0, 1]  # dg -> C1
    ]
    
    col_offset = 0
    
    # Block 1: cols 0,1
    for t in range(7):
        U[:, col_offset] = strassen_U[t]
        # V indices for cols 0,1: 0,1,9,10
        V[0, col_offset] = strassen_V[t][0]
        V[1, col_offset] = strassen_V[t][1]
        V[9, col_offset] = strassen_V[t][2]
        V[10, col_offset] = strassen_V[t][3]
        # W indices for cols 0,1: 0,1,2,3
        W[0, col_offset] = strassen_W[t][0]
        W[1, col_offset] = strassen_W[t][1]
        W[2, col_offset] = strassen_W[t][2]
        W[3, col_offset] = strassen_W[t][3]
        col_offset += 1
        
    # Block 2: cols 2,3
    for t in range(7):
        U[:, col_offset] = strassen_U[t]
        # V indices for cols 2,3: 2,3,11,12
        V[2, col_offset] = strassen_V[t][0]
        V[3, col_offset] = strassen_V[t][1]
        V[11, col_offset] = strassen_V[t][2]
        V[12, col_offset] = strassen_V[t][3]
        # W indices for cols 2,3: 4,5,6,7
        W[4, col_offset] = strassen_W[t][0]
        W[5, col_offset] = strassen_W[t][1]
        W[6, col_offset] = strassen_W[t][2]
        W[7, col_offset] = strassen_W[t][3]
        col_offset += 1
        
    # Block 3: cols 4,5
    for t in range(7):
        U[:, col_offset] = strassen_U[t]
        # V indices for cols 4,5: 4,5,13,14
        V[4, col_offset] = strassen_V[t][0]
        V[5, col_offset] = strassen_V[t][1]
        V[13, col_offset] = strassen_V[t][2]
        V[14, col_offset] = strassen_V[t][3]
        # W indices for cols 4,5: 8,9,10,11
        W[8, col_offset] = strassen_W[t][0]
        W[9, col_offset] = strassen_W[t][1]
        W[10, col_offset] = strassen_W[t][2]
        W[11, col_offset] = strassen_W[t][3]
        col_offset += 1
        
    # Block 4: cols 6,7
    for t in range(7):
        U[:, col_offset] = strassen_U[t]
        # V indices for cols 6,7: 6,7,15,16
        V[6, col_offset] = strassen_V[t][0]
        V[7, col_offset] = strassen_V[t][1]
        V[15, col_offset] = strassen_V[t][2]
        V[16, col_offset] = strassen_V[t][3]
        # W indices for cols 6,7: 12,13,14,15
        W[12, col_offset] = strassen_W[t][0]
        W[13, col_offset] = strassen_W[t][1]
        W[14, col_offset] = strassen_W[t][2]
        W[15, col_offset] = strassen_W[t][3]
        col_offset += 1
        
    # Block 5: col 8
    for t in range(4):
        U[:, col_offset] = school_U[t]
        # V indices for col 8: 8, 17
        V[8, col_offset] = school_V[t][0]
        V[17, col_offset] = school_V[t][1]
        # W indices for col 8: 16, 17
        W[16, col_offset] = school_W[t][0]
        W[17, col_offset] = school_W[t][1]
        col_offset += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
