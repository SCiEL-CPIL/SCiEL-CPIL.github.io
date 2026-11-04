"""Matrix multiplication 3x4x7 rank-minimal decomposition using block Strassen."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strassen decomposition for 2x2x2 subproblems
    strassen_decomp = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # M3
        ([0,0,0,1], [0,0,1,-1], [1,1,0,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # M7
    ]
    
    def add_strassen(i0, j0, k0, rank_offset):
        for a_vec, b_vec, c_vec in strassen_decomp:
            for idx in range(4):
                if a_vec[idx] != 0:
                    i_loc = idx // 2
                    j_loc = idx % 2
                    U[(i0 + i_loc) * m + (j0 + j_loc)][rank_offset] += a_vec[idx]
            for idx in range(4):
                if b_vec[idx] != 0:
                    j_loc = idx // 2
                    k_loc = idx % 2
                    V[(j0 + j_loc) * p + (k0 + k_loc)][rank_offset] += b_vec[idx]
            for idx in range(4):
                if c_vec[idx] != 0:
                    i_loc = idx // 2
                    k_loc = idx % 2
                    W[(k0 + k_loc) * n + (i0 + i_loc)][rank_offset] += c_vec[idx]
            rank_offset += 1
        return rank_offset
    
    def add_schoolbook(i0, j0, k0, ni, nj, nk, rank_offset):
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    U[(i0 + i) * m + (j0 + j)][rank_offset] = 1
                    V[(j0 + j) * p + (k0 + k)][rank_offset] = 1
                    W[(k0 + k) * n + (i0 + i)][rank_offset] = 1
                    rank_offset += 1
        return rank_offset
    
    # Manually design the decomposition for 3x4x7:
    # Partition: rows (2,1), cols of A/rows of B (2,2), cols of B/cols of C (4,3)
    # 
    # Block 1: A[0:2,0:2] * B[0:2,0:4] -> 2x2x4
    #   Split B cols into (2,2): 
    #     B[0:2,0:2] -> 2x2x2, use Strassen (7)
    #     B[0:2,2:4] -> 2x2x2, use Strassen (7)
    #   Total: 14
    #
    # Block 2: A[0:2,2:4] * B[2:4,0:4] -> 2x2x4
    #   Split B cols into (2,2):
    #     B[2:4,0:2] -> 2x2x2, use Strassen (7)
    #     B[2:4,2:4] -> 2x2x2, use Strassen (7)
    #   Total: 14
    #
    # Block 3: A[0:2,0:2] * B[0:2,4:7] -> 2x2x3
    #   Split B cols into (2,1):
    #     B[0:2,4:6] -> 2x2x2, use Strassen (7)
    #     B[0:2,6:7] -> 2x2x1, schoolbook (4)
    #   Total: 11
    #
    # Block 4: A[0:2,2:4] * B[2:4,4:7] -> 2x2x3
    #   Split B cols into (2,1):
    #     B[2:4,4:6] -> 2x2x2, use Strassen (7)
    #     B[2:4,6:7] -> 2x2x1, schoolbook (4)
    #   Total: 11
    #
    # Block 5: A[2:3,0:2] * B[0:2,0:4] -> 1x2x4
    #   Schoolbook: 1*2*4 = 8
    #
    # Block 6: A[2:3,2:4] * B[2:4,0:4] -> 1x2x4
    #   Schoolbook: 1*2*4 = 8
    #
    # Block 7: A[2:3,0:2] * B[0:2,4:7] -> 1x2x3
    #   Schoolbook: 1*2*3 = 6
    #
    # Block 8: A[2:3,2:4] * B[2:4,4:7] -> 1x2x3
    #   Schoolbook: 1*2*3 = 6
    #
    # Total: 14+14+11+11+8+8+6+6 = 78
    
    total_rank = 78
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank = 0
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> Strassen
    rank = add_strassen(0, 0, 0, rank)  # 7
    # Block 1b: A[0:2,0:2] * B[0:2,2:4] -> Strassen
    rank = add_strassen(0, 0, 2, rank)  # 7
    
    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> Strassen
    rank = add_strassen(0, 2, 0, rank)  # 7
    # Block 2b: A[0:2,2:4] * B[2:4,2:4] -> Strassen
    rank = add_strassen(0, 2, 2, rank)  # 7
    
    # Block 3: A[0:2,0:2] * B[0:2,4:6] -> Strassen
    rank = add_strassen(0, 0, 4, rank)  # 7
    # Block 3b: A[0:2,0:2] * B[0:2,6:7] -> Schoolbook 2x2x1
    rank = add_schoolbook(0, 0, 6, 2, 2, 1, rank)  # 4
    
    # Block 4: A[0:2,2:4] * B[2:4,4:6] -> Strassen
    rank = add_strassen(0, 2, 4, rank)  # 7
    # Block 4b: A[0:2,2:4] * B[2:4,6:7] -> Schoolbook 2x2x1
    rank = add_schoolbook(0, 2, 6, 2, 2, 1, rank)  # 4
    
    # Block 5: A[2:3,0:2] * B[0:2,0:4] -> Schoolbook 1x2x4
    rank = add_schoolbook(2, 0, 0, 1, 2, 4, rank)  # 8
    
    # Block 6: A[2:3,2:4] * B[2:4,0:4] -> Schoolbook 1x2x4
    rank = add_schoolbook(2, 2, 0, 1, 2, 4, rank)  # 8
    
    # Block 7: A[2:3,0:2] * B[0:2,4:7] -> Schoolbook 1x2x3
    rank = add_schoolbook(2, 0, 4, 1, 2, 3, rank)  # 6
    
    # Block 8: A[2:3,2:4] * B[2:4,4:7] -> Schoolbook 1x2x3
    rank = add_schoolbook(2, 2, 4, 1, 2, 3, rank)  # 6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
