import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen for 4x4x4
    # 2x2 Strassen matrices
    U_2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W_2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # 4x4 Strassen via Kronecker product
    U_4 = np.kron(U_2, U_2)
    V_4 = np.kron(V_2, V_2)
    W_4 = np.kron(W_2, W_2)
    
    # 2. Schoolbook for 4x4x5
    # n=4, m=4, p_part=5
    p_part = 5
    rank_sb = n * m * p_part
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p_part, rank_sb))
    W_sb = np.zeros((p_part*n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p_part):
                U_sb[i*m + j, col] = 1
                V_sb[j*p_part + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
                
    # 3. Combine
    # V and W need padding to match total dimensions
    # Total m*p = 4*9 = 36
    # V_4 is 16x49, V_sb is 20x80
    # We need V_4 to be 36x49 (pad bottom 20 rows with 0)
    # We need V_sb to be 36x80 (pad top 16 rows with 0)
    
    V_4_padded = np.zeros((m*p, U_4.shape[1]))
    V_4_padded[:m*4, :] = V_4
    
    V_sb_padded = np.zeros((m*p, U_sb.shape[1]))
    V_sb_padded[m*4:, :] = V_sb
    
    # W_4 is 16x49, W_sb is 20x80
    # We need W_4 to be 36x49 (pad bottom 20 rows with 0)
    # We need W_sb to be 36x80 (pad top 16 rows with 0)
    
    W_4_padded = np.zeros((p*n, W_4.shape[1]))
    W_4_padded[:4*n, :] = W_4
    
    W_sb_padded = np.zeros((p*n, W_sb.shape[1]))
    W_sb_padded[4*n:, :] = W_sb
    
    U_final = np.hstack([U_4, U_sb])
    V_final = np.hstack([V_4_padded, V_sb_padded])
    W_final = np.hstack([W_4_padded, W_sb_padded])
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
