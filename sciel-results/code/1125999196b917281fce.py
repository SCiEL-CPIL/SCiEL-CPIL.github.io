import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    
    # Target rank 56
    # Structure:
    # A (2x4) = [A0, A1] where A0, A1 are 2x2
    # B (4x8) = [B0; B1] where B0, B1 are 2x8
    # C = A0 B0 + A1 B1
    # For each alpha in {0, 1}:
    #   B_alpha (2x8) is split into 4 blocks of 2x2: B_{alpha, 0}, B_{alpha, 1}, B_{alpha, 2}, B_{alpha, 3}
    #   C_alpha (2x4) is split into 4 blocks of 2x2: C_{alpha, 0}, ..., C_{alpha, 3}
    #   C_{alpha, beta} = A_alpha B_{alpha, beta}
    #   Each 2x2x2 product uses Strassen (rank 7)
    # Total 8 products * 7 = 56.
    
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen coefficients for 2x2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A B
    # M1 = a11 (b11 + b22)
    # M2 = (a11 + a22) b22
    # M3 = (a21 + a22) b11
    # M4 = a22 (b21 + b12)
    # M5 = a11 (b12 + b22)
    # M6 = (a11 + a12) b21
    # M7 = (a21 + a11) (b12 + b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the linear forms for U (on A), V (on B), and W (on C) for one 2x2x2 block.
    # Indices for A in 2x2: 0:11, 1:12, 2:21, 3:22
    # Indices for B in 2x2: 0:11, 1:12, 2:21, 3:22
    # Indices for C in 2x2: 0:11, 1:12, 2:21, 3:22
    
    # U forms (7 vectors of size 4)
    u_forms = [
        [1, 0, 0, 0],   # a11
        [1, 0, 1, 1],   # a11+a22
        [0, 0, 1, 1],   # a21+a22
        [0, 0, 0, 1],   # a22
        [1, 0, 0, 0],   # a11
        [1, 1, 0, 0],   # a11+a12
        [1, 0, 1, 0]    # a21+a11
    ]
    
    # V forms (7 vectors of size 4)
    v_forms = [
        [1, 0, 0, 1],   # b11+b22
        [0, 0, 0, 1],   # b22
        [1, 0, 0, 0],   # b11
        [0, 1, 1, 0],   # b21+b12
        [0, 1, 0, 1],   # b12+b22
        [0, 0, 1, 0],   # b21
        [0, 1, 0, 1]    # b12+b22
    ]
    
    # W forms (7 vectors of size 4)
    # Coefficients for M1..M7 in C11, C12, C21, C22
    # C11: 1, 0, 0, 0, -1, 0, 1
    # C12: 0, 0, 1, 0, 1, 0, 0
    # C21: 0, 1, 0, 1, 0, 0, 0
    # C22: 1, -1, 1, 0, 0, 1, 0
    w_forms = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [0, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    # Mapping of global indices
    # A is 2x4. Row i, Col j. Index = i*m + j.
    # A0 is cols 0,1. A1 is cols 2,3.
    # A_alpha local index l (0..3) corresponds to global index:
    # if alpha=0: i in {0,1}, j in {0,1}. Local l = i*2 + j. Global = i*4 + j.
    # if alpha=1: i in {0,1}, j in {2,3}. Local l = i*2 + (j-2). Global = i*4 + j.
    
    # B is 4x8. Row j, Col k. Index = j*p + k.
    # B0 is rows 0,1. B1 is rows 2,3.
    # B_{alpha, beta} is 2x2 block.
    # alpha in {0,1} -> row offset 2*alpha.
    # beta in {0,1,2,3} -> col offset 2*beta.
    # Local l (0..3) = i*2 + j where i,j in 0..1.
    # Global row = 2*alpha + i.
    # Global col = 2*beta + j.
    # Global index = (2*alpha + i)*8 + (2*beta + j).
    
    # C is 2x8. Row i, Col k. Index = k*n + i (trace convention W[k*n+i]).
    # C_{alpha, beta} is 2x2 block.
    # alpha in {0,1} -> This corresponds to the sum term index? No.
    # C = C0 + C1.
    # C0 = A0 B0. C1 = A1 B1.
    # The output C is the sum.
    # So for a given output block (i, beta), it receives contributions from alpha=0 and alpha=1.
    # Wait, C_{i, k} = (C0)_{i,k} + (C1)_{i,k}.
    # So we just add the W terms.
    
    # We iterate over alpha (0,1), beta (0,1,2,3), and strassen_id (0..6).
    # Total 2 * 4 * 7 = 56 columns.
    
    col = 0
    for alpha in range(2):
        for beta in range(4):
            for s in range(7):
                # Fill U
                # A_alpha local indices 0..3
                for local_i in range(2):
                    for local_j in range(2):
                        l = local_i * 2 + local_j
                        global_i = local_i
                        global_j = 2 * alpha + local_j
                        global_idx_A = global_i * m + global_j
                        U[global_idx_A][col] += u_forms[s][l]
                        
                # Fill V
                # B_{alpha, beta} local indices 0..3
                for local_i in range(2):
                    for local_j in range(2):
                        l = local_i * 2 + local_j
                        global_row = 2 * alpha + local_i
                        global_col = 2 * beta + local_j
                        global_idx_B = global_row * p + global_col
                        V[global_idx_B][col] += v_forms[s][l]
                        
                # Fill W
                # C_{alpha, beta} local indices 0..3
                # C is 2x8.
                # C_{i, k} = sum_alpha (A_alpha B_alpha)_{i,k}
                # For a specific alpha and beta, the block is at rows 0,1 and cols 2*beta, 2*beta+1.
                for local_i in range(2):
                    for local_j in range(2):
                        l = local_i * 2 + local_j
                        global_i = local_i
                        global_k = 2 * beta + local_j
                        global_idx_C = global_k * n + global_i
                        W[global_idx_C][col] += w_forms[s][l]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
