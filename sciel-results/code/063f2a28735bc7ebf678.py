import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    rank_core = 49
    rank_strip1 = 32
    rank_strip2 = 1
    total_rank = rank_core + rank_strip1 + rank_strip2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 factors (4x7 matrices)
    # Rows correspond to A00, A01, A10, A11
    U2 = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 0]
    ]
    V2 = [
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, -1, -1, 1, 0, 0]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Coefficients for 4x4 Strassen
    # alpha[r][I][J]
    alpha = [
        [[1, 0], [0, 1]], # r=0
        [[0, 0], [1, 1]], # r=1
        [[1, 0], [0, 0]], # r=2
        [[0, 0], [0, 1]], # r=3
        [[1, 1], [0, 0]], # r=4
        [[-1, 0], [1, 0]],# r=5
        [[0, 1], [0, -1]] # r=6
    ]
    
    # beta[r][J][K]
    beta = [
        [[1, 0], [0, 1]], # r=0
        [[1, 0], [0, 0]], # r=1
        [[0, 1], [0, -1]],# r=2
        [[-1, 1], [0, 0]],# r=3
        [[0, 0], [0, 1]], # r=4
        [[1, 1], [0, 0]], # r=5
        [[0, 1], [0, 1]]  # r=6
    ]
    
    # gamma[r][I][K]
    gamma = [
        [[1, 0], [0, 1]], # r=0
        [[0, 0], [1, -1]],# r=1
        [[0, 1], [0, 1]], # r=2
        [[1, 0], [1, 0]], # r=3
        [[-1, 1], [0, 0]],# r=4
        [[0, 0], [0, 1]], # r=5
        [[1, 0], [0, 0]]  # r=6
    ]
    
    # Fill Block 1
    for r in range(7):
        for s in range(7):
            for I in range(2):
                for J in range(2):
                    a_val = alpha[r][I][J]
                    if a_val != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                i = 2 * I + i_prime
                                j = 2 * J + j_prime
                                u_idx = 2 * i_prime + j_prime
                                U[i * m + j][col] += a_val * U2[u_idx][s]
                                
            for J in range(2):
                for K in range(2):
                    b_val = beta[r][J][K]
                    if b_val != 0:
                        for j_prime in range(2):
                            for k_prime in range(2):
                                j = 2 * J + j_prime
                                k = 2 * K + k_prime
                                v_idx = 2 * j_prime + k_prime
                                V[j * p + k][col] += b_val * V2[v_idx][s]
                                
            for I in range(2):
                for K in range(2):
                    c_val = gamma[r][I][K]
                    if c_val != 0:
                        for i_prime in range(2):
                            for k_prime in range(2):
                                i = 2 * I + i_prime
                                k = 2 * K + k_prime
                                w_idx = 2 * k_prime + i_prime
                                W[k * n + i][col] += c_val * W2[w_idx][s]
                                
            col += 1
            
    # Fill Block 2
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Fill Block 3
    for i in range(4):
        U[i * m + 4][col] = 1
    for k in range(6):
        V[4 * p + k][col] = 1
    for i in range(4):
        for k in range(6):
            W[k * n + i][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
