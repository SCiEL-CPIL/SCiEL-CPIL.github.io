"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Block size 3, 2x2 blocks
    block_size = 3
    num_blocks = 2
    
    # Strassen coefficients for 2x2 block multiplication
    # alpha[k][i][j] for A blocks
    # beta[k][i][j] for B blocks
    # gamma[k][a][b] for C blocks
    # k is 0-indexed 0..6 corresponding to M1..M7
    
    alphas = [
        [[1, 0], [0, 1]],  # M1: A00 + A11
        [[0, 0], [1, 1]],  # M2: A10 + A11
        [[1, 0], [0, 0]],  # M3: A00
        [[0, 0], [0, 1]],  # M4: A11
        [[1, 1], [0, 0]],  # M5: A00 + A01
        [[-1, 0], [1, 0]], # M6: A10 - A00
        [[0, 1], [0, -1]], # M7: A01 - A11
    ]
    
    betas = [
        [[1, 0], [0, 1]],  # M1: B00 + B11
        [[1, 0], [0, 0]],  # M2: B00
        [[0, 1], [0, -1]], # M3: B01 - B11
        [[-1, 0], [1, 0]], # M4: B10 - B00
        [[0, 0], [0, 1]],  # M5: B11
        [[1, 1], [0, 0]],  # M6: B00 + B01
        [[0, 1], [0, 1]],  # M7: B10 + B11
    ]
    
    gammas = [
        [[1, 0], [0, 1]],  # M1 -> C00, C11
        [[0, 0], [1, 0]],  # M2 -> C10
        [[0, 1], [0, 0]],  # M3 -> C01
        [[1, 0], [1, 0]],  # M4 -> C00, C10
        [[-1, 1], [0, 0]], # M5 -> C00(-1), C01(1)
        [[0, 0], [0, 1]],  # M6 -> C11
        [[1, 0], [0, 1]],  # M7 -> C00, C11
    ]
    
    rank = 7 * (block_size ** 3)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(7):
        # Iterate over 3x3 schoolbook terms for the block multiplication
        # r, c, s are 0..2
        for r in range(block_size):
            for c in range(block_size):
                for s in range(block_size):
                    # Construct U, V, W vectors for this rank-1 term
                    # U index: row = 3*i + r, col = 3*j + s
                    # V index: row = 3*i + s, col = 3*j + c
                    # W index: row = 3*a + r, col = 3*b + c
                    
                    for i in range(num_blocks):
                        for j in range(num_blocks):
                            val = alphas[k][i][j]
                            if val != 0:
                                u_row = i * block_size + r
                                u_col = j * block_size + s
                                U[u_row * m + u_col][col] = val
                            
                            val = betas[k][i][j]
                            if val != 0:
                                v_row = i * block_size + s
                                v_col = j * block_size + c
                                V[v_row * p + v_col][col] = val
                            
                    for a in range(num_blocks):
                        for b in range(num_blocks):
                            val = gammas[k][a][b]
                            if val != 0:
                                w_row = a * block_size + r
                                w_col = b * block_size + c
                                W[w_row * n + w_col][col] = val
                                
                    col += 1
                    
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
