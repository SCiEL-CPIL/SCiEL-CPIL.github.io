"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen 2x2 matrices
    # U_S[2i+j, r]
    U_S = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # V_S[2j+k, r]
    V_S = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    # W_S[2k+i, r]
    W_S = [
        [1, 0, 0, -1, 0, -1, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, -1, 0, 1]
    ]
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for i in range(4):
        for j in range(4):
            i1, i0 = i // 2, i % 2
            j1, j0 = j // 2, j % 2
            row_u = 4 * i + j
            row_u_s1 = 2 * i1 + j1
            row_u_s0 = 2 * i0 + j0
            
            for k in range(4):
                j1, j0 = j // 2, j % 2
                k1, k0 = k // 2, k % 2
                row_v = 4 * j + k
                row_v_s1 = 2 * j1 + k1
                row_v_s0 = 2 * j0 + k0
                
                i1, i0 = i // 2, i % 2
                k1, k0 = k // 2, k % 2
                row_w = 4 * k + i
                row_w_s1 = 2 * k1 + i1
                row_w_s0 = 2 * k0 + i0
                
                for r in range(7):
                    for s in range(7):
                        col = 7 * r + s
                        val_u = U_S[row_u_s1][r] * U_S[row_u_s0][s]
                        val_v = V_S[row_v_s1][r] * V_S[row_v_s0][s]
                        val_w = W_S[row_w_s1][r] * W_S[row_w_s0][s]
                        
                        U[row_u][col] = val_u
                        V[row_v][col] = val_v
                        W[row_w][col] = val_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
