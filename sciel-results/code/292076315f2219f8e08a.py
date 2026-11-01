import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 14
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen U, V, W for 2x2x2
    # U_s, V_s are vectors of length 4 (flattened 2x2)
    # P_s are 2x2 matrices for W_s
    
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    P_strassen = [
        np.array([[1, 0], [0, 1]]), # M1
        np.array([[0, 0], [1, -1]]),# M2
        np.array([[0, 1], [0, 1]]), # M3
        np.array([[1, 0], [1, 0]]), # M4
        np.array([[-1, 1], [0, 0]]),# M5
        np.array([[0, 0], [0, 1]]), # M6
        np.array([[1, 0], [0, 0]])  # M7
    ]
    
    for s in range(7):
        # Block 1 (k in 0,1)
        r1 = s
        # U
        U[:, r1] = U_strassen[s]
        
        # V
        # V_strassen[s] is flattened j*2+k' for j in 0,1 and k' in 0,1
        # V index is j*4+k. For block 1, k=k'.
        for j in range(2):
            for kp in range(2):
                V[j * 4 + kp, r1] = V_strassen[s][j * 2 + kp]
                
        # W
        # W index is k*2+i. For block 1, k=k'.
        # W[k*2+i, r] = P_strassen[s][i, k]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, r1] = P_strassen[s][i, k]
                
        # Block 2 (k in 2,3)
        r2 = 7 + s
        # U
        U[:, r2] = U_strassen[s]
        
        # V
        # V index is j*4+k. For block 2, k=k'+2.
        for j in range(2):
            for kp in range(2):
                V[j * 4 + (kp + 2), r2] = V_strassen[s][j * 2 + kp]
                
        # W
        # W index is k*2+i. For block 2, k=k'+2.
        for k in range(2):
            for i in range(2):
                W[(k + 2) * 2 + i, r2] = P_strassen[s][i, k]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
