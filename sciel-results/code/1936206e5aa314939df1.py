import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    # We split B into 3 blocks of 3x3.
    # For each block, we compute A (2x3) * B_block (3x3).
    # Split A into A_left (2x2) and A_right (2x1).
    # Split B_block into B_top (2x3) and B_bottom (1x3).
    # A * B_block = A_left * B_top + A_right * B_bottom.
    # A_left * B_top is 2x2 by 2x3. Schoolbook rank 12.
    # A_right * B_bottom is 2x1 by 1x3. Rank 1.
    # Total per block: 13.
    # Total rank: 3 * 13 = 39.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)] # 6 x 39
    V = [[0] * rank for _ in range(m * p)] # 27 x 39
    W = [[0] * rank for _ in range(p * n)] # 18 x 39
    
    col = 0
    for t in range(3):
        # Block t corresponds to columns 3t, 3t+1, 3t+2 of B
        # Part 1: j in {0, 1} (first 2 rows of B, first 2 cols of A)
        # A_left is A[:, 0:2], B_top is B[0:2, :]
        # Schoolbook for 2x2 by 2x3
        for i in range(2):
            for j in range(2):
                for k_local in range(3):
                    k_global = 3 * t + k_local
                    # U[i, col] = 1
                    U[i][col] = 1
                    # V[j * p + k_global, col] = 1
                    V[j * p + k_global][col] = 1
                    # W[k_global * n + i, col] = 1
                    W[k_global * n + i][col] = 1
                    col += 1
        
        # Part 2: j = 2 (last row of B, last col of A)
        # A_right is A[:, 2], B_bottom is B[2, :]
        # Rank 1 term
        for i in range(2):
            U[i][col] = 1
        for k_local in range(3):
            k_global = 3 * t + k_local
            V[2 * p + k_global][col] = 1
        # W part: w_j = 1 for j=2
        # The term is sum_i sum_k u_i v_k w_2 e_i e_k e_2
        # So W[2, col] = 1
        W[2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
