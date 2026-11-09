"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition that splits the middle dimension m=4 into two m=2 blocks.
    # R(M_{3,4,7}) <= 2 * R(M_{3,2,7})
    # R(M_{3,2,7}) <= R(M_{3,2,3}) + R(M_{3,2,4})
    # R(M_{3,2,3}) = 12 (3 * R(M_{1,2,1})? No, 3 * 2 = 6? Wait.
    # Let's trace carefully.
    
    # Rank of M_{1,2,1} is 2.
    # Rank of M_{1,2,2} is 2?
    # C = A1 B1 + A2 B2. Rank 2.
    # So R(M_{3,2,1}) = 3 * 2 = 6.
    # R(M_{3,2,2}) = 3 * 2 = 6.
    # R(M_{3,2,3}) = R(M_{3,2,2}) + R(M_{3,2,1}) = 6 + 6 = 12.
    # R(M_{3,2,4}) = R(M_{3,2,2}) + R(M_{3,2,2}) = 6 + 6 = 12.
    # R(M_{3,2,7}) = 12 + 12 = 24.
    # Total Rank = 2 * 24 = 48.
    
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a block M_{n, 2, p_sub}
    def add_block(a_start, b_start, p_sub, rank_offset):
        nonlocal col
        # a_start: index of first row of A block (0 or 2)
        # b_start: index of first row of B block (0 or 2)
        # p_sub: number of columns in B block
        # rank_offset: not used, we use col
        
        # Decompose M_{3, 2, p_sub}
        # Split p_sub into chunks of size 1 or 2?
        # We know R(M_{3,2,1})=6, R(M_{3,2,2})=6.
        # So we can just iterate k in p_sub and use R(M_{3,2,1}) logic?
        # R(M_{3,2,1}) = 6.
        # For each k in range(p_sub):
        #   Add 6 terms for column k.
        
        for k in range(p_sub):
            # We need to compute C[:, k] = A_block * B_block[:, k]
            # A_block is 3x2. B_col is 2x1.
            # C_i = A_i0 B_0 + A_i1 B_1
            # Terms:
            # 1. A_i0 B_0 -> C_i
            # 2. A_i1 B_1 -> C_i
            # For each i in 0..2:
            for i in range(n):
                # Term 1: A[i+a_start, 0] * B[0+b_start, k]
                # U index: (i+a_start)*m + 0 ? No, U is (n*m, rank).
                # U row corresponds to A[i, j]. Index i*m + j.
                # Here j=0.
                u_idx = (i + a_start) * m + 0
                v_idx = (0 + b_start) * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
                # Term 2: A[i+a_start, 1] * B[1+b_start, k]
                u_idx = (i + a_start) * m + 1
                v_idx = (1 + b_start) * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    # Block 1: A rows 0-1, B rows 0-1, C cols 0-6 (p=7)
    add_block(0, 0, 7, 0)
    
    # Block 2: A rows 2-3, B rows 2-3, C cols 0-6 (p=7)
    add_block(2, 2, 7, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
