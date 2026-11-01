import numpy as np

def construct():
    n, m, p = 2, 2, 15
    
    # Define Strassen decomposition for p=2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices for U: 0:00, 1:01, 2:10, 3:11
    # Indices for V: 0:00, 1:01, 2:10, 3:11
    # Indices for W: 0:00, 1:10, 2:01, 3:11 (Note: W index k*n+i)
    # Wait, my previous W mapping:
    # W_0 = C_00 (k=0, i=0)
    # W_1 = C_10 (k=0, i=1)
    # W_2 = C_01 (k=1, i=0)
    # W_3 = C_11 (k=1, i=1)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=1: p1 = (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=2: p2 = (c+d)e
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=3: p3 = a(f-h)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=4: p4 = d(g-e)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=5: p5 = (a+b)h
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=6: p6 = (c-a)(e+f)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=7: p7 = (b-d)(g+h)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Define Schoolbook decomposition for p=1
    # U shape (4, 4), V shape (2, 4), W shape (2, 4)
    # V indices: 0:00, 1:10
    # W indices: 0:00, 1:10
    
    U1 = np.zeros((4, 4))
    V1 = np.zeros((2, 4))
    W1 = np.zeros((2, 4))
    
    for r in range(4):
        i = r // 2
        j = r % 2
        U1[i*m + j, r] = 1
        V1[j, r] = 1
        W1[i, r] = 1
        
    # Global Matrices
    rank = 7 * 7 + 4 # 53
    U = np.zeros((4, rank))
    V = np.zeros((30, rank))
    W = np.zeros((30, rank))
    
    col_offset = 0
    # Place 7 blocks of p=2
    for t in range(7):
        # U
        U[:, col_offset:col_offset+7] = U2
        
        # V
        # Local V rows: 0,1,2,3
        # Global V rows:
        # 0 -> 2t
        # 1 -> 2t+1
        # 2 -> 15+2t
        # 3 -> 15+2t+1
        v_rows = [2*t, 2*t+1, 15+2*t, 15+2*t+1]
        for local_r, global_r in enumerate(v_rows):
            V[global_r, col_offset:col_offset+7] = V2[local_r, :]
            
        # W
        # Local W rows: 0,1,2,3
        # Global W rows:
        # 0 -> 4t
        # 1 -> 4t+1
        # 2 -> 4t+2
        # 3 -> 4t+3
        w_rows = [4*t, 4*t+1, 4*t+2, 4*t+3]
        for local_r, global_r in enumerate(w_rows):
            W[global_r, col_offset:col_offset+7] = W2[local_r, :]
            
        col_offset += 7
        
    # Place 1 block of p=1
    # U
    U[:, col_offset:col_offset+4] = U1
    
    # V
    # Local V rows: 0, 1
    # Global V rows:
    # 0 -> 14
    # 1 -> 29
    v_rows = [14, 29]
    for local_r, global_r in enumerate(v_rows):
        V[global_r, col_offset:col_offset+4] = V1[local_r, :]
        
    # W
    # Local W rows: 0, 1
    # Global W rows:
    # 0 -> 28
    # 1 -> 29
    w_rows = [28, 29]
    for local_r, global_r in enumerate(w_rows):
        W[global_r, col_offset:col_offset+4] = W1[local_r, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
