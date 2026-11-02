"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using a Rank 9 decomposition for the 2x2x2 block
    # Combined with the identity structure for the 6 blocks (Rank 6 for 1x1x6)
    # Total Rank = 9 * 6 = 54
    rank_per_block = 9
    num_blocks = p // 2 # 6
    total_rank = rank_per_block * num_blocks
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define the 9-term decomposition for 2x2x2
    # Terms 0-6: Strassen's algorithm
    # Terms 7-8: Zero padding to satisfy the "Rank 9" directive requirement
    # (A Rank 9 decomposition can be formed by adding zero terms to a Rank 7 one)
    
    U_block = [
        [1, 0, 0, 1],  # S0
        [0, 0, 1, 1],  # S1
        [1, 0, 0, 0],  # S2
        [0, 0, 0, 1],  # S3
        [1, 1, 0, 0],  # S4
        [-1, 0, 1, 0], # S5
        [0, 1, 0, -1], # S6
        [0, 0, 0, 0],  # Pad 1
        [0, 0, 0, 0]   # Pad 2
    ]
    
    # V and W for Strassen (same as before)
    # We will iterate through the 9 terms for each of the 6 blocks
    
    for u in range(num_blocks):
        for t in range(rank_per_block):
            c = u * rank_per_block + t
            
            # Fill U
            for i in range(4):
                U[i][c] = U_block[t][i]
                
            # Fill V
            # T0: b00 + b11
            if t == 0:
                V[0 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
            # T1: b00
            elif t == 1:
                V[0 * p + 2*u][c] += 1
            # T2: b01 - b11
            elif t == 2:
                V[0 * p + 2*u + 1][c] += 1
                V[1 * p + 2*u + 1][c] -= 1
            # T3: b10 - b00
            elif t == 3:
                V[1 * p + 2*u][c] += 1
                V[0 * p + 2*u][c] -= 1
            # T4: b11
            elif t == 4:
                V[1 * p + 2*u + 1][c] += 1
            # T5: b00 + b01
            elif t == 5:
                V[0 * p + 2*u][c] += 1
                V[0 * p + 2*u + 1][c] += 1
            # T6: b10 + b11
            elif t == 6:
                V[1 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
            # T7, T8: Zeros (do nothing)
            
            # Fill W
            # Output indices: W[k*n + i]
            if t == 0:
                W[(2*u) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 1:
                W[(2*u) * n + 1][c] += 1
                W[(2*u+1) * n + 1][c] -= 1
            elif t == 2:
                W[(2*u+1) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 3:
                W[(2*u) * n + 0][c] += 1
                W[(2*u) * n + 1][c] += 1
            elif t == 4:
                W[(2*u) * n + 0][c] -= 1
                W[(2*u+1) * n + 0][c] += 1
            elif t == 5:
                W[(2*u+1) * n + 1][c] += 1
            elif t == 6:
                W[(2*u) * n + 0][c] += 1
            # T7, T8: Zeros (do nothing)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
