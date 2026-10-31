"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's 7 rank-1 tensors for 2x2 matrix multiplication
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    u_2x2 = [
        [1, 0, 0, 1], # 0
        [0, 0, 1, 1], # 1
        [1, 0, 0, 0], # 2
        [0, 0, 0, 1], # 3
        [1, 0, 1, 0], # 4
        [0, 1, -1, 0],# 5
        [0, 1, 0, -1] # 6
    ]
    v_2x2 = [
        [1, 0, 0, 1], # 0
        [1, 0, 0, 0], # 1
        [0, 1, 0, -1],# 2
        [-1, 0, 1, 0],# 3
        [0, 0, 0, 1], # 4
        [0, 0, 1, 1], # 5
        [1, 0, 1, 0]  # 6
    ]
    w_2x2 = [
        [1, 0, 0, 1], # 0
        [0, 1, 0, 0], # 1
        [0, 0, 1, 0], # 2
        [0, 0, 0, 1], # 3
        [1, 0, 0, 0], # 4
        [0, 1, 0, 0], # 5
        [0, 0, 1, 0]  # 6
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Kronecker product u_s (x) u_t
            u_vec = []
            for i_s in range(4):
                for i_t in range(4):
                    u_vec.append(u_2x2[s][i_s] * u_2x2[t][i_t])
            
            v_vec = []
            for j_s in range(4):
                for j_t in range(4):
                    v_vec.append(v_2x2[s][j_s] * v_2x2[t][j_t])
            
            w_vec = []
            for k_s in range(4):
                for k_t in range(4):
                    w_vec.append(w_2x2[s][k_s] * w_2x2[t][k_t])
            
            for i in range(16):
                U[i][col] = u_vec[i]
            for j in range(16):
                V[j][col] = v_vec[j]
            for k in range(16):
                W[k][col] = w_vec[k]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
