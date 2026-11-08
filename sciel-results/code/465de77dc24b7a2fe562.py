"""Matrix multiplication 3,4,8 using rank-88 decomposition with deadline guard."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8
    for j in range(4):
        for k in range(8):
            if time.time() > deadline:
                break
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
        if time.time() > deadline:
            break
    
    # Part 2: i=1,2, Strassen on 2x2 blocks
    strassen_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]  
    ]
    strassen_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    strassen_W = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    for j_block in range(2):
        if time.time() > deadline:
            break
        j_start = j_block * 2
        for k_block in range(4):
            if time.time() > deadline:
                break
            k_start = k_block * 2
            for t in range(7):
                if time.time() > deadline:
                    break
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += strassen_U[t][l]
                            
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += strassen_V[t][l]
                            
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i][col] += strassen_W[t][l]
                            
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
