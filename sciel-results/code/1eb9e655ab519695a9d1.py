import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2x2
    # (alpha, beta, gamma)
    # alpha: A blocks, beta: B blocks, gamma: C blocks
    # Indices (i, j) for 2x2 blocks
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Part 1: 4x4x4 (cols 0-3 of B)
    # Rank 49
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = s * 7 + s2
            
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 (cols 4-5 of B)
    # Rank 28
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: Top (rows 0-1), Bot (rows 2-3)
    # C blocks: Top (rows 0-1), Bot (rows 2-3)
    
    # Multiplications:
    # C_Top = A00 * B_Top + A01 * B_Bot
    # C_Bot = A10 * B_Top + A11 * B_Bot
    
    mults = [
        ((0,0), 0), # A00, B_Top -> C_Top
        ((0,1), 1), # A01, B_Bot -> C_Top
        ((1,0), 0), # A10, B_Top -> C_Bot
        ((1,1), 1)  # A11, B_Bot -> C_Bot
    ]
    
    col_start_2 = 49
    
    for m_idx, (a_block, b_row_block) in enumerate(mults):
        c_row_block = a_block[0] # C row block matches A row block
        
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = col_start_2 + m_idx * 7 + s2
            
            for (si, sj), val_a2 in alpha_2.items():
                r = (a_block[0] * 2 + si) * m + (a_block[1] * 2 + sj)
                U[r][col] += val_a2
                
            for (si, sj), val_b2 in beta_2.items():
                r = (b_row_block * 2 + si) * p + (4 + sj)
                V[r][col] += val_b2
                
            for (si, sj), val_c2 in gamma_2.items():
                r = (4 + sj) * n + (c_row_block * 2 + si)
                W[r][col] += val_c2

    return {"U": U, "V": V, "W": W}
