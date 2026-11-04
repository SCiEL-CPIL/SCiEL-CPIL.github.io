"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # U_old[i*2+j, r], V_old[j*2+k, r], W_old[k*2+i, r]
    U_old = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, -1, -1]  # A11
    ]
    V_old = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    W_old = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    rank_2x2 = 7
    rank_4x4 = rank_2x2 * rank_2x2 # 49
    
    U = [[0] * rank_4x4 for _ in range(n * m)]
    V = [[0] * rank_4x4 for _ in range(m * p)]
    W = [[0] * rank_4x4 for _ in range(p * n)]
    
    for I in range(n):
        for J in range(m):
            i = I // 2
            i_prime = I % 2
            j = J // 2
            j_prime = J % 2
            
            for r1 in range(rank_2x2):
                for r2 in range(rank_2x2):
                    R = r1 * rank_2x2 + r2
                    val = U_old[i * 2 + j][r1] * U_old[i_prime * 2 + j_prime][r2]
                    if val != 0:
                        U[I * m + J][R] += val
                        
    for J in range(m):
        for K in range(p):
            j = J // 2
            j_prime = J % 2
            k = K // 2
            k_prime = K % 2
            
            for r1 in range(rank_2x2):
                for r2 in range(rank_2x2):
                    R = r1 * rank_2x2 + r2
                    val = V_old[j * 2 + k][r1] * V_old[j_prime * 2 + k_prime][r2]
                    if val != 0:
                        V[J * p + K][R] += val
                        
    for K in range(p):
        for I in range(n):
            k = K // 2
            k_prime = K % 2
            i = I // 2
            i_prime = I % 2
            
            for r1 in range(rank_2x2):
                for r2 in range(rank_2x2):
                    R = r1 * rank_2x2 + r2
                    val = W_old[k * 2 + i][r1] * W_old[k_prime * 2 + i_prime][r2]
                    if val != 0:
                        W[K * n + I][R] += val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
