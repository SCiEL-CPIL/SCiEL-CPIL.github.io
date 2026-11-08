import time

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2
    # U_s is length 4 (i=0,1; j=0,1 -> index i*2+j)
    # V_s is length 4 (j=0,1; k=0,1 -> index j*2+k)
    # W_s is length 4 (k=0,1; i=0,1 -> index k*2+i)
    
    # r=0
    U0 = [1, 0, 0, 1]
    V0 = [1, 0, 0, 1]
    W0 = [1, 0, 0, 1]
    
    # r=1
    U1 = [0, 0, 1, 1]
    V1 = [1, 0, 0, 0]
    W1 = [0, 0, 1, 0]
    
    # r=2
    U2 = [1, 0, 0, 0]
    V2 = [0, 1, 0, -1]
    W2 = [0, 1, 0, 0]
    
    # r=3
    U3 = [0, 0, 0, 1]
    V3 = [-1, 1, 0, 0]
    W3 = [0, 0, 1, 0]
    
    # r=4
    U4 = [1, 1, 0, 0]
    V4 = [0, 1, 0, 0]
    W4 = [-1, 1, 0, 0]
    
    # r=5
    U5 = [-1, 0, 1, 0]
    V5 = [1, 1, 0, 0]
    W5 = [0, 0, 0, 1]
    
    # r=6
    U6 = [0, 1, 0, -1]
    V6 = [0, 0, 1, 1]
    W6 = [1, 0, 0, 0]
    
    Strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    Strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    Strassen_W = [W0, W1, W2, W3, W4, W5, W6]
    
    # Blocks:
    # 0: A1 B11 (i 0-1, j 0-1, k 0-1)
    # 1: A1 B12 (i 0-1, j 0-1, k 2-3)
    # 2: A2 B21 (i 0-1, j 2-3, k 0-1)
    # 3: A2 B22 (i 0-1, j 2-3, k 2-3)
    
    for b in range(4):
        j_offset = 2 * (b // 2)
        k_offset = 2 * (b % 2)
        
        for s in range(7):
            r = 7 * b + s
            u_vec = Strassen_U[s]
            v_vec = Strassen_V[s]
            w_vec = Strassen_W[s]
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    global_i = i
                    global_j = j + j_offset
                    U[global_i * m + global_j][r] = u_vec[local_idx]
                    
            # Fill V
            for j in range(2):
                for k in range(2):
                    local_idx = j * 2 + k
                    global_j = j + j_offset
                    global_k = k + k_offset
                    V[global_j * p + global_k][r] = v_vec[local_idx]
                    
            # Fill W
            for k in range(2):
                for i in range(2):
                    local_idx = k * 2 + i
                    global_k = k + k_offset
                    global_i = i
                    W[global_k * n + global_i][r] = w_vec[local_idx]
                    
    return {"U": U, "V": V, "W": W}
