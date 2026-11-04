"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition of rank n*m*p
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, column] = 1
                V_fb[j * p + k, column] = 1
                W_fb[k * n + i, column] = 1
                column += 1
                
    # Attempt Rank 41 decomposition
    # Part 1: 3x3x3 (Laderman, Rank 23)
    # Part 2: 3x3x2 (Schoolbook, Rank 18)
    
    rank_attempt = 23 + 18
    U_att = np.zeros((n * m, rank_attempt))
    V_att = np.zeros((m * p, rank_attempt))
    W_att = np.zeros((p * n, rank_attempt))
    
    # Laderman 3x3x3 Terms
    # Format: (U_indices, V_indices, W_indices)
    # U index: i*m + j
    # V index: j*3 + k (for 3x3x3 part)
    # W index: k*n + i
    
    laderman_terms = [
        ([ (0, 1) ], [ (0, 1) ], [ (0, 1) ]), # 0
        ([ (1, 1) ], [ (3, 1) ], [ (0, 1) ]), # 1: A01 B10 -> U[1], V[3], W[0]
        ([ (2, 1) ], [ (6, 1) ], [ (0, 1) ]), # 2: A02 B20 -> U[2], V[6], W[0]
        ([ (3, 1) ], [ (1, 1) ], [ (1, 1) ]), # 3: A10 B01 -> U[3], V[1], W[1]
        ([ (4, 1) ], [ (4, 1) ], [ (4, 1) ]), # 4: A11 B11 -> U[4], V[4], W[4]
        ([ (5, 1) ], [ (7, 1) ], [ (4, 1) ]), # 5: A12 B21 -> U[5], V[7], W[4]
        ([ (6, 1) ], [ (2, 1) ], [ (2, 1) ]), # 6: A20 B02 -> U[6], V[2], W[2]
        ([ (7, 1) ], [ (5, 1) ], [ (5, 1) ]), # 7: A21 B12 -> U[7], V[5], W[5]
        ([ (8, 1) ], [ (8, 1) ], [ (8, 1) ]), # 8: A22 B22 -> U[8], V[8], W[8]
        ([ (0, 1), (4, 1) ], [ (4, 1), (8, 1) ], [ (4, 1), (8, 1) ]), # 9
        ([ (1, 1), (2, 1) ], [ (6, 1), (3, 1) ], [ (0, 1), (1, 1) ]), # 10
        ([ (3, 1), (6, 1) ], [ (2, 1), (1, 1) ], [ (2, 1), (1, 1) ]), # 11
        ([ (4, 1), (8, 1) ], [ (8, 1), (4, 1) ], [ (8, 1), (4, 1) ]), # 12
        ([ (2, 1), (5, 1) ], [ (3, 1), (6, 1) ], [ (0, 1), (4, 1) ]), # 13
        ([ (6, 1), (7, 1) ], [ (1, 1), (5, 1) ], [ (1, 1), (5, 1) ]), # 14
        ([ (5, 1), (8, 1) ], [ (7, 1), (8, 1) ], [ (4, 1), (8, 1) ]), # 15
        ([ (0, 1), (6, 1) ], [ (2, 1), (0, 1) ], [ (2, 1), (0, 1) ]), # 16
        ([ (1, 1), (7, 1) ], [ (5, 1), (3, 1) ], [ (5, 1), (3, 1) ]), # 17
        ([ (2, 1), (3, 1) ], [ (7, 1), (1, 1) ], [ (4, 1), (1, 1) ]), # 18
        ([ (3, 1), (4, 1) ], [ (5, 1), (4, 1) ], [ (5, 1), (4, 1) ]), # 19
        ([ (6, 1), (8, 1) ], [ (6, 1), (8, 1) ], [ (6, 1), (8, 1) ]), # 20
        ([ (0, 1), (8, 1) ], [ (8, 1), (0, 1) ], [ (8, 1), (0, 1) ]), # 21
        # Need 1 more term? Laderman has 23.
    ]
    
    # The above hardcoded list is likely incorrect in details.
    # I will skip the hardcoded Laderman and return Rank 45 to ensure validity.
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
