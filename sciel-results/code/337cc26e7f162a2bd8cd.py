"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 48
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We construct the decomposition by splitting the p=8 dimension into two p=4 blocks.
    # C[:, 0:4] = A B[:, 0:4]
    # C[:, 4:8] = A B[:, 4:8]
    # Each is a 2x4x4 multiplication.
    # We assume Rank(2x4x4) = 24.
    # We need a rank-24 decomposition for 2x4x4.
    # 2x4x4 can be split into two 2x4x2 multiplications.
    # C[:, 0:2] = A B[:, 0:2]
    # C[:, 2:4] = A B[:, 2:4]
    # We assume Rank(2x4x2) = 12.
    
    # Let's try to build Rank(2x4x2) = 12.
    # A is 2x4, B is 4x2.
    # Split A into A1 (2x2, cols 0,1) and A2 (2x2, cols 2,3).
    # Split B into B1 (2x2, rows 0,1) and B2 (2x2, rows 2,3).
    # C = A1 B1 + A2 B2.
    # This is sum of two 2x2x2 products.
    # Rank(2x2x2) = 7.
    # Total rank 14.
    # Can we do 12?
    # If not, we might be stuck at 56 or 48 (if 24 is possible).
    # If Rank(2x4x2) is 14, then Rank(2x4x4) is 28, Rank(2x4x8) is 56.
    # So 56 is likely optimal if 2x4x2 is 14.
    # Is 2x4x2 rank 12 possible?
    # Lower bound for 2x4x2 is 8.
    # Let's try to find a rank-12 decomposition for 2x4x2.
    
    # If we can't find it, we return the rank-56 solution.
    
    # For now, return the rank-56 solution which is known to work.
    # The current program already does this.
    # I will keep the rank-56 logic but optimize the code.
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    current_rank = 0
    
    # Group 1: A1 * Btop (rows 0,1 of B)
    # A1 is rows 0,1 of A? No, A is 2x4.
    # The previous code split A into two 2x2 blocks (cols 0,1 and 2,3).
    # And B into 4 blocks of 2x2 (rows 0,1; 2,3; etc).
    # This yields 4 * 7 = 28 terms for A1, and 28 for A2. Total 56.
    
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    U[i * m + j_loc][r] = u_loc[i * 2 + j_loc]
            for j_loc in range(2):
                for k_loc in range(2):
                    V[j_loc * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
            for i in range(2):
                for k_loc in range(2):
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            current_rank += 1
            
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    U[i * m + (j_loc + 2)][r] = u_loc[i * 2 + j_loc]
            for j_loc in range(2):
                for k_loc in range(2):
                    V[(j_loc + 2) * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
            for i in range(2):
                for k_loc in range(2):
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            current_rank += 1
            
    U_list = [[int(U[i, r]) for r in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, r]) for r in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, r]) for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
