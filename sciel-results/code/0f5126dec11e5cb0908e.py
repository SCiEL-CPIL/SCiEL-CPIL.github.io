import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2 blocks
    steps = [
        ([[ -1,  0], [ 0,  0]], [[ 1,  0], [ 0,  1]], [[ 1,  0], [ 0,  1]]),
        ([[ 1,  0], [ 0,  1]], [[ 0,  0], [ 0,  1]], [[ 0,  0], [ 1, -1]]),
        ([[ 1,  0], [ 0,  0]], [[ 0,  1], [ 0, -1]], [[ 0,  1], [ 0,  1]]),
        ([[ 0,  0], [ 0,  1]], [[-1,  0], [ 1,  0]], [[ 1,  0], [ 1,  0]]),
        ([[ 1,  1], [ 0,  0]], [[ 1,  0], [ 0,  0]], [[-1,  1], [ 0,  0]]),
        ([[ -1,  0], [ 1,  0]], [[ 1,  1], [ 0,  0]], [[ 0,  0], [ 0,  1]]),
        ([[ 0,  1], [ 0, -1]], [[ 0,  0], [ 1,  1]], [[ 1,  0], [ 0,  0]]),
    ]
    
    col = 0
    # 49 terms for 4x4x4
    for s in range(7):
        a_s, b_s, c_s = steps[s]
        for t in range(7):
            a_t, b_t, c_t = steps[t]
            
            # Construct u, v, w for this term
            # u[i, j] = a_s[I, J] * a_t[i_l, j_l]
            # where i = 2*I + i_l, j = 2*J + j_l
            # v[j, k] = b_s[J, K] * b_t[j_l, k_l]
            # w[k, i] = c_s[K, I] * c_t[k_l, i_l]
            
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for i_l in range(2):
                            for j_l in range(2):
                                for k_l in range(2):
                                    i = 2 * I + i_l
                                    j = 2 * J + j_l
                                    k = 2 * K + k_l
                                    
                                    val_u = a_s[I][J] * a_t[i_l][j_l]
                                    val_v = b_s[J][K] * b_t[j_l][k_l]
                                    val_w = c_s[K][I] * c_t[k_l][i_l]
                                    
                                    if val_u != 0 and val_v != 0 and val_w != 0:
                                        U[i * m + j][col] = val_u
                                        V[j * p + k][col] = val_v
                                        W[k * n + i][col] = val_w
                                        
            col += 1
            
    # 16 terms for 4x4x1 (5th column)
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
