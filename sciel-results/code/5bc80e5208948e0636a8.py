import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen factors for 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # a (0,0)
        [0, 0, 0, 0, 1, 0, 1],  # b (0,1)
        [0, 1, 0, 0, 0, 1, 0],  # c (1,0)
        [1, 1, 0, 1, 0, 0, -1]  # d (1,1)
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # e (0,0)
        [0, 0, 1, 0, 0, 1, 0],  # f (0,1)
        [0, 0, 0, 1, 0, 0, 1],  # g (1,0)
        [1, 0, -1, 0, 1, 0, 1]  # h (1,1)
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c11 (0,0)
        [-1, 0, 1, 0, 1, 1, 0],  # c12 (0,1)
        [0, 1, 1, 1, 0, 0, -1],  # c21 (1,0)
        [1, 1, -1, 1, 0, 0, 0]   # c22 (1,1)
    ])
    
    # Kronecker products give raw factors
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Initialize final matrices
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Map U rows
    # Raw index r = r_top * 4 + r_bot
    # r_top corresponds to A_top (i_top, j_top) -> r_top = i_top * 2 + j_top
    # r_bot corresponds to A_bot (i_bot, j_bot) -> r_bot = i_bot * 2 + j_bot
    # Target index u_idx = i * 4 + j where i = i_top * 2 + i_bot, j = j_top * 2 + j_bot
    for i_top in range(2):
        for j_top in range(2):
            r_top = i_top * 2 + j_top
            for i_bot in range(2):
                for j_bot in range(2):
                    r_bot = i_bot * 2 + j_bot
                    r_raw = r_top * 4 + r_bot
                    
                    i = i_top * 2 + i_bot
                    j = j_top * 2 + j_bot
                    u_idx = i * 4 + j
                    
                    U[u_idx, :] = U4_raw[r_raw, :]
                    
    # Map V rows
    # Raw index s = s_top * 4 + s_bot
    # s_top corresponds to B_top (j_top, k_top) -> s_top = j_top * 2 + k_top
    # s_bot corresponds to B_bot (j_bot, k_bot) -> s_bot = j_bot * 2 + k_bot
    # Target index v_idx = j * 4 + k where j = j_top * 2 + j_bot, k = k_top * 2 + k_bot
    for j_top in range(2):
        for k_top in range(2):
            s_top = j_top * 2 + k_top
            for j_bot in range(2):
                for k_bot in range(2):
                    s_bot = j_bot * 2 + k_bot
                    s_raw = s_top * 4 + s_bot
                    
                    j = j_top * 2 + j_bot
                    k = k_top * 2 + k_bot
                    v_idx = j * 4 + k
                    
                    V[v_idx, :] = V4_raw[s_raw, :]
                    
    # Map W rows
    # Raw index t = t_top * 4 + t_bot
    # t_top corresponds to C_top (i_top, k_top) -> t_top = i_top * 2 + k_top
    # t_bot corresponds to C_bot (i_bot, k_bot) -> t_bot = i_bot * 2 + k_bot
    # Target index w_idx = k * 4 + i where k = k_top * 2 + k_bot, i = i_top * 2 + i_bot
    for i_top in range(2):
        for k_top in range(2):
            t_top = i_top * 2 + k_top
            for i_bot in range(2):
                for k_bot in range(2):
                    t_bot = i_bot * 2 + k_bot
                    t_raw = t_top * 4 + t_bot
                    
                    i = i_top * 2 + i_bot
                    k = k_top * 2 + k_bot
                    w_idx = k * 4 + i
                    
                    W[w_idx, :] = W4_raw[t_raw, :]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
