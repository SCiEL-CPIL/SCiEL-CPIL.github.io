"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    
    # Rank 49 for 4x4x4 (Strassen) + Rank 48 for 4x4x3 (Schoolbook)
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 vectors (u, v, w)
    # Indexing: 00, 01, 10, 11
    s_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    for t in range(7):
        for s in range(7):
            # Expand 2x2 vectors to 4x4
            u4 = [0] * 16
            v4 = [0] * 16
            w4 = [0] * 16
            
            for I in range(2):
                for J in range(2):
                    outer_u = s_u[t][I * 2 + J]
                    outer_v = s_v[t][I * 2 + J]
                    outer_w = s_w[t][I * 2 + J]
                    
                    for i in range(2):
                        for j in range(2):
                            inner_u = s_u[s][i * 2 + j]
                            inner_v = s_v[s][i * 2 + j]
                            inner_w = s_w[s][i * 2 + j]
                            
                            idx = (2 * I + i) * 4 + (2 * J + j)
                            u4[idx] = outer_u * inner_u
                            v4[idx] = outer_v * inner_v
                            w4[idx] = outer_w * inner_w
                            
            for idx in range(16):
                if u4[idx] != 0:
                    U[idx][col] = u4[idx]
                if v4[idx] != 0:
                    V[idx][col] = v4[idx]
                if w4[idx] != 0:
                    # W index is k*n + i. w4 index is i*n + k.
                    # i = idx // 4, k = idx % 4
                    i_global = idx // 4
                    k_global = idx % 4
                    w_idx = k_global * 4 + i_global
                    W[w_idx][col] = w4[idx]
            col += 1
            
    # Schoolbook for 4x4x3
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
