"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen matrices for 2x2 multiplication
    # U_s[i, j], V_s[j, k], W_s[k, i]
    # Indices are 0-based
    U_strassen = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, -1], [1, 0]]
    ]
    V_strassen = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, -1]],
        [[0, 0], [0, 1]],
        [[0, 0], [0, 1]],
        [[1, 0], [1, 0]],
        [[0, 1], [0, 1]]
    ]
    W_strassen = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    col = 0
    for b in range(6):
        # b=0,1,2: A1 * B1, B2, B3
        # b=3,4,5: A2 * B4, B5, B6
        a_block = b // 3  # 0 for A1, 1 for A2
        b_block = b % 3   # 0, 1, 2 for B col blocks
        
        for s in range(7):
            for i_prime in range(2):
                for j_prime in range(2):
                    # Global i
                    i = i_prime
                    # Global j
                    j = 2 * a_block + j_prime
                    # U index
                    u_idx = i * m + j
                    U[u_idx][col] = U_strassen[s][i_prime][j_prime]
                    
            for j_prime in range(2):
                for k_prime in range(2):
                    # Global j
                    j = 2 * a_block + j_prime
                    # Global k
                    k = 2 * b_block + k_prime
                    # V index
                    v_idx = j * p + k
                    V[v_idx][col] = V_strassen[s][j_prime][k_prime]
                    
            for k_prime in range(2):
                for i_prime in range(2):
                    # Global k
                    k = 2 * b_block + k_prime
                    # Global i
                    i = i_prime
                    # W index
                    w_idx = k * n + i
                    W[w_idx][col] = W_strassen[s][k_prime][i_prime]
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
