import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen matrices for 2x2x2
    # U_str[i', j', s]
    U_str = [
        [1, 0, 1, 0, 1, -1, 0], # i'=0, j'=0
        [0, 0, 0, 0, 0, 0, 1], # i'=0, j'=1
        [0, 1, 0, 0, 1, 0, 0], # i'=1, j'=0
        [1, 1, 0, 1, 0, 1, -1]  # i'=1, j'=1
    ]
    
    # V_str[j', k', s]
    V_str = [
        [1, 1, 0, -1, 0, 1, 0], # j'=0, k'=0
        [0, 0, 1, 0, 0, 0, 0], # j'=0, k'=1
        [0, 0, 0, 1, 0, 1, 1], # j'=1, k'=0
        [1, 0, -1, 0, 1, 0, 1]  # j'=1, k'=1
    ]
    
    # W_str[k', i', s]
    W_str = [
        [1, 0, 0, 1, -1, 0, 1], # k'=0, i'=0
        [0, 0, 1, 0, 1, 0, 0], # k'=0, i'=1
        [0, 1, 0, 1, 0, 0, 0], # k'=1, i'=0
        [1, -1, 1, 0, 0, 1, 0]  # k'=1, i'=1
    ]
    
    # 8 blocks
    # Block b maps local (i', j', k') to global (i, j, k)
    # b = 0: j in 0,1; k in 0,1
    # b = 1: j in 0,1; k in 2,3
    # b = 2: j in 0,1; k in 4,5
    # b = 3: j in 0,1; k in 6,7
    # b = 4: j in 2,3; k in 0,1
    # b = 5: j in 2,3; k in 2,3
    # b = 6: j in 2,3; k in 4,5
    # b = 7: j in 2,3; k in 6,7
    
    for b in range(8):
        col_offset = b * 7
        # Determine j offset and k offset
        j_offset = (b // 4) * 2
        k_offset = (b % 4) * 2
        
        for s in range(7):
            col = col_offset + s
            for i_prime in range(2):
                for j_prime in range(2):
                    i = i_prime
                    j = j_offset + j_prime
                    u_val = U_str[i_prime * 2 + j_prime][s]
                    U[i * m + j][col] += u_val
                    
                    v_val = V_str[j_prime * 2 + 0][s] # Wait, V_str index is j', k'
                    # Need to loop k_prime too for V and W
                    
            # Better to loop all local indices for each s
            
        # Redo loop structure
        for s in range(7):
            col = col_offset + s
            for i_prime in range(2):
                for j_prime in range(2):
                    i = i_prime
                    j = j_offset + j_prime
                    U[i * m + j][col] += U_str[i_prime * 2 + j_prime][s]
                    
            for j_prime in range(2):
                for k_prime in range(2):
                    j = j_offset + j_prime
                    k = k_offset + k_prime
                    V[j * p + k][col] += V_str[j_prime * 2 + k_prime][s]
                    
            for k_prime in range(2):
                for i_prime in range(2):
                    i = i_prime
                    k = k_offset + k_prime
                    W[k * n + i][col] += W_str[k_prime * 2 + i_prime][s]
                    
    return {"U": U, "V": V, "W": W}

