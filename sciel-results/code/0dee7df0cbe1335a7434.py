import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    rank = 35
    U = [[0] * rank for _ in range(n * m)] # 4 x 35
    V = [[0] * rank for _ in range(m * p)] # 20 x 35
    W = [[0] * rank for _ in range(p * n)] # 20 x 35
    
    # Strassen factors for 2x2 multiplication
    # U_vec: 4 entries (i*2+j)
    # V_vec: 4 entries (j*2+k)
    # W_vec: 4 entries (k*2+i)
    
    # Product 1
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1]
    W1 = [1, 0, 0, 1]
    
    # Product 2
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    W2 = [0, 1, 0, -1]
    
    # Product 3
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    W3 = [0, 0, 1, 1]
    
    # Product 4
    U4 = [0, 0, 0, 1]
    V4 = [-1, 0, 1, 0]
    W4 = [1, 1, 0, 0]
    
    # Product 5
    U5 = [1, 1, 0, 0]
    V5 = [0, 0, 0, 1]
    W5 = [-1, 0, 1, 0]
    
    # Product 6
    U6 = [-1, 0, 1, 0]
    V6 = [1, 1, 0, 0]
    W6 = [0, 0, 0, 1]
    
    # Product 7
    U7 = [0, 1, 0, -1]
    V7 = [0, 0, 1, 1]
    W7 = [1, 0, 0, 0]
    
    factors = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    col = 0
    for q in range(5):
        for Uf, Vf, Wf in factors:
            # Fill U
            for i in range(2):
                for j in range(2):
                    idx_u = i * 2 + j
                    U[idx_u][col] = Uf[idx_u]
            
            # Fill V
            for j in range(2):
                for k in range(2):
                    idx_v_local = j * 2 + k
                    idx_v_global = j * 10 + (2 * q + k)
                    V[idx_v_global][col] = Vf[idx_v_local]
            
            # Fill W
            for k in range(2):
                for i in range(2):
                    idx_w_local = k * 2 + i
                    idx_w_global = (2 * q + k) * 2 + i
                    W[idx_w_global][col] = Wf[idx_w_local]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
