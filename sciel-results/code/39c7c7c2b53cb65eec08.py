import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Start time for budget management
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    # Rank is 98 (49 for left 4x4, 49 for right 4x4)
    rank = 98
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen factors for 2x2x2
    # U2[i*2+j, k] for i,j in 0..1, k in 0..6
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ]
    
    # Define Block Coefficients A_b, B_b, C_b for 7 Strassen products
    # Index b in 0..6
    # A_b is 2x2, B_b is 2x2, C_b is 2x2
    A_blocks = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    
    B_blocks = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[-1, 0], [1, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    
    C_blocks = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]],# M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    # Fill U, V, W
    for r in range(rank):
        if r < 49:
            r_block = r // 7
            r_scalar = r % 7
            is_left = True
            r_offset = 0
        else:
            r_prime = r - 49
            r_block = r_prime // 7
            r_scalar = r_prime % 7
            is_left = False
            r_offset = 49 # Not used for index, just logic
            
        A_b = A_blocks[r_block]
        B_b = B_blocks[r_block]
        C_b = C_blocks[r_block]
        
        for i in range(4):
            for j in range(4):
                I = i // 2
                J = j // 2
                i_prime = i % 2
                j_prime = j % 2
                
                val_U = A_b[I][J] * U2[i_prime * 2 + j_prime][r_scalar]
                U[i * 4 + j][r] = val_U
                
                # V and W depend on k
                # For Left part (r < 49), k is 0..3
                # For Right part (r >= 49), k is 4..7
                
        for j in range(4):
            for k in range(8):
                if is_left:
                    if k >= 4:
                        V[j * 8 + k][r] = 0
                        W[k * 4 + 0][r] = 0 # Placeholder, will overwrite in W loop
                        continue
                    K = k // 2
                    k_prime = k % 2
                else:
                    if k < 4:
                        V[j * 8 + k][r] = 0
                        continue
                    K = (k - 4) // 2
                    k_prime = (k - 4) % 2
                
                J = j // 2
                J_prime = j % 2
                
                val_V = B_b[J][K] * V2[J_prime * 2 + k_prime][r_scalar]
                V[j * 8 + k][r] = val_V
                
        for k in range(8):
            for i in range(4):
                if is_left:
                    if k >= 4:
                        W[k * 4 + i][r] = 0
                        continue
                    K = k // 2
                    k_prime = k % 2
                else:
                    if k < 4:
                        W[k * 4 + i][r] = 0
                        continue
                    K = (k - 4) // 2
                    k_prime = (k - 4) % 2
                
                I = i // 2
                i_prime = i % 2
                
                val_W = C_b[K][I] * W2[k_prime * 2 + i_prime][r_scalar]
                W[k * 4 + i][r] = val_W

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
