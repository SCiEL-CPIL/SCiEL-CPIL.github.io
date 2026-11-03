import time
import sys

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use a block decomposition.
    # A is 3x5, B is 5x5, C is 3x5.
    # Split B into 2x2, 2x3, 3x2, 3x3 blocks.
    # Split A into 3x2, 3x3 blocks.
    
    # Sub-problem 1: A1 (3x2) * B11 (2x2) -> C11 (3x2)
    # Sub-problem 2: A2 (3x3) * B21 (3x2) -> C12 (3x2)
    # Sub-problem 3: A1 (3x2) * B12 (2x3) -> C21 (3x3)
    # Sub-problem 4: A2 (3x3) * B22 (3x3) -> C22 (3x3)
    
    # Ranks (Naive):
    # 1: 3x2x2 -> rank 12
    # 2: 3x3x2 -> rank 18
    # 3: 3x2x3 -> rank 18
    # 4: 3x3x3 -> rank 27
    # Total rank: 12 + 18 + 18 + 27 = 75
    
    rank_1 = 12
    rank_2 = 18
    rank_3 = 18
    rank_4 = 27
    rank = rank_1 + rank_2 + rank_3 + rank_4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a decomposition
    def add_decomp(n_sub, m_sub, p_sub, U_sub, V_sub, W_sub, i_offset, j_offset, k_offset, sub_rank):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for c in range(sub_rank):
                    if U_sub[i*m_sub+j][c] != 0:
                        U[(i + i_offset) * m + (j + j_offset)][col + c] += U_sub[i*m_sub+j][c]
        for j in range(m_sub):
            for k in range(p_sub):
                for c in range(sub_rank):
                    if V_sub[j*p_sub+k][c] != 0:
                        V[(j + j_offset) * p + (k + k_offset)][col + c] += V_sub[j*p_sub+k][c]
        for k in range(p_sub):
            for i in range(n_sub):
                for c in range(sub_rank):
                    if W_sub[k*n_sub+i][c] != 0:
                        W[(k + k_offset) * n + (i + i_offset)][col + c] += W_sub[k*n_sub+i][c]
        col += sub_rank

    # 1. 3x2x2 (rank 12)
    U1 = [[0]*rank_1 for _ in range(3*2)]
    V1 = [[0]*rank_1 for _ in range(2*2)]
    W1 = [[0]*rank_1 for _ in range(2*3)]
    c = 0
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U1[i*2+j][c] = 1
                V1[j*2+k][c] = 1
                W1[k*3+i][c] = 1
                c += 1
    add_decomp(3, 2, 2, U1, V1, W1, 0, 0, 0, rank_1)
    
    # 2. 3x3x2 (rank 18)
    U2 = [[0]*rank_2 for _ in range(3*3)]
    V2 = [[0]*rank_2 for _ in range(3*2)]
    W2 = [[0]*rank_2 for _ in range(2*3)]
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U2[i*3+j][c] = 1
                V2[j*2+k][c] = 1
                W2[k*3+i][c] = 1
                c += 1
    add_decomp(3, 3, 2, U2, V2, W2, 0, 2, 0, rank_2)
    
    # 3. 3x2x3 (rank 18)
    U3 = [[0]*rank_3 for _ in range(3*2)]
    V3 = [[0]*rank_3 for _ in range(2*3)]
    W3 = [[0]*rank_3 for _ in range(3*3)]
    c = 0
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U3[i*2+j][c] = 1
                V3[j*3+k][c] = 1
                W3[k*3+i][c] = 1
                c += 1
    add_decomp(3, 2, 3, U3, V3, W3, 0, 0, 2, rank_3)
    
    # 4. 3x3x3 (rank 27)
    U4 = [[0]*rank_4 for _ in range(3*3)]
    V4 = [[0]*rank_4 for _ in range(3*3)]
    W4 = [[0]*rank_4 for _ in range(3*3)]
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U4[i*3+j][c] = 1
                V4[j*3+k][c] = 1
                W4[k*3+i][c] = 1
                c += 1
    add_decomp(3, 3, 3, U4, V4, W4, 0, 2, 2, rank_4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
