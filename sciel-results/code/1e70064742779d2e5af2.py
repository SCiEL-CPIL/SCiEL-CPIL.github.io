import numpy as np

def strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
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
    W[2, 3] = 1; W[3, 3] = -1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    
    return U, V, W

def strassen_4x4():
    U2, V2, W2 = strassen_2x2()
    
    U_block = np.zeros((4, 7))
    V_block = np.zeros((4, 7))
    W_block = np.zeros((4, 7))
    
    U_block[0, 0] = 1; U_block[3, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1
    
    U_block[2, 1] = 1; U_block[3, 1] = 1
    V_block[0, 1] = 1
    W_block[2, 1] = 1; W_block[3, 1] = -1
    
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1; W_block[3, 2] = 1
    
    U_block[3, 3] = 1
    V_block[2, 3] = 1; V_block[0, 3] = -1
    W_block[2, 3] = 1; W_block[3, 3] = -1
    
    U_block[0, 4] = 1; U_block[1, 4] = 1
    V_block[3, 4] = 1
    W_block[0, 4] = -1; W_block[3, 4] = 1
    
    U_block[2, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1
    
    U_block[1, 6] = 1; U_block[3, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = 1; W_block[1, 6] = -1
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for b in range(4):
        for p in range(7):
            coeff = U_block[b, p]
            if coeff != 0:
                U[b*4:(b+1)*4, p*7:(p+1)*7] = coeff * U2
            coeff_v = V_block[b, p]
            if coeff_v != 0:
                V[b*4:(b+1)*4, p*7:(p+1)*7] = coeff_v * V2
            coeff_w = W_block[b, p]
            if coeff_w != 0:
                W[b*4:(b+1)*4, p*7:(p+1)*7] = coeff_w * W2
                
    return U, V, W

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140
    
    U4, V4, W4 = strassen_4x4()
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    U[:, :49] = U4
    V[:16, :49] = V4
    W[:16, :49] = W4
    
    U[:, 49:] = U4
    V[16:32, 49:] = V4
    W[16:32, 49:] = W4
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
