import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """
    Returns U, V, W for schoolbook decomposition of n x m by m x p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """
    Returns U, V, W for Strassen decomposition of 2x2 by 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4 = A11(B00-B10)
    U[3, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose 3x3x5 into blocks.
    # Partition indices 0,1,2 into {0,1} and {2}.
    # Partition columns 0..4 into {0,1} and {2,3,4}.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2)
    # Block 2: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] (2x1x2)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5] (2x2x3)
    # Block 4: A[0:2, 2] * B[2, 2:5] -> C[0:2, 2:5] (2x1x3)
    # Block 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] (1x2x2)
    # Block 6: A[2, 2] * B[2, 0:2] -> C[2, 0:2] (1x1x2)
    # Block 7: A[2, 0:2] * B[0:2, 2:5] -> C[2, 2:5] (1x2x3)
    # Block 8: A[2, 2] * B[2, 2:5] -> C[2, 2:5] (1x1x3)
    
    # Total Rank = 7 + 4 + 12 + 6 + 4 + 2 + 6 + 3 = 44
    
    total_rank = 44
    U = np.zeros((9, total_rank))
    V = np.zeros((15, total_rank))
    W = np.zeros((15, total_rank))
    
    col_offset = 0
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, i_start, j_start, k_start, i_len, j_len, k_len):
        nonlocal col_offset
        rank_block = U_block.shape[1]
        for c in range(rank_block):
            # U
            for i in range(i_len):
                for j in range(j_len):
                    U[i_start + i, col_offset + c] += U_block[i * j_len + j, c] * (1 if i < i_len and j < j_len else 0)
                    # Wait, U_block indices are local.
                    # U_block[i_local, j_local, c]
                    # Global U index: (i_start + i_local) * 3 + (j_start + j_local)
            # This is getting complicated to map manually.
            pass
    
    # Let's just build the full matrices by iterating over the blocks and their terms.
    
    # Block 1: 2x2x2 Strassen
    U1, V1, W1 = _strassen_2x2()
    for c in range(7):
        # Map local (i, j) to global (i, j)
        # i in 0,1 -> global i in 0,1
        # j in 0,1 -> global j in 0,1
        for i_l in range(2):
            for j_l in range(2):
                U[i_l * 3 + j_l, col_offset + c] += U1[i_l * 2 + j_l, c]
        for j_l in range(2):
            for k_l in range(2):
                V[j_l * 5 + k_l, col_offset + c] += V1[j_l * 2 + k_l, c]
        for k_l in range(2):
            for i_l in range(2):
                W[k_l * 3 + i_l, col_offset + c] += W1[k_l * 2 + i_l, c]
    col_offset += 7
    
    # Block 2: 2x1x2 (Schoolbook)
    U2, V2, W2 = _schoolbook(2, 1, 2)
    for c in range(4):
        for i_l in range(2):
            for j_l in range(1):
                U[i_l * 3 + (2 + j_l), col_offset + c] += U2[i_l * 1 + j_l, c]
        for j_l in range(1):
            for k_l in range(2):
                V[(2 + j_l) * 5 + k_l, col_offset + c] += V2[j_l * 2 + k_l, c]
        for k_l in range(2):
            for i_l in range(2):
                W[k_l * 3 + i_l, col_offset + c] += W2[k_l * 2 + i_l, c]
    col_offset += 4
    
    # Block 3: 2x2x3 (Schoolbook)
    U3, V3, W3 = _schoolbook(2, 2, 3)
    for c in range(12):
        for i_l in range(2):
            for j_l in range(2):
                U[i_l * 3 + j_l, col_offset + c] += U3[i_l * 2 + j_l, c]
        for j_l in range(2):
            for k_l in range(3):
                V[j_l * 5 + (2 + k_l), col_offset + c] += V3[j_l * 3 + k_l, c]
        for k_l in range(3):
            for i_l in range(2):
                W[(2 + k_l) * 3 + i_l, col_offset + c] += W3[k_l * 2 + i_l, c]
    col_offset += 12
    
    # Block 4: 2x1x3 (Schoolbook)
    U4, V4, W4 = _schoolbook(2, 1, 3)
    for c in range(6):
        for i_l in range(2):
            for j_l in range(1):
                U[i_l * 3 + (2 + j_l), col_offset + c] += U4[i_l * 1 + j_l, c]
        for j_l in range(1):
            for k_l in range(3):
                V[(2 + j_l) * 5 + (2 + k_l), col_offset + c] += V4[j_l * 3 + k_l, c]
        for k_l in range(3):
            for i_l in range(2):
                W[(2 + k_l) * 3 + i_l, col_offset + c] += W4[k_l * 2 + i_l, c]
    col_offset += 6
    
    # Block 5: 1x2x2 (Schoolbook)
    U5, V5, W5 = _schoolbook(1, 2, 2)
    for c in range(4):
        for i_l in range(1):
            for j_l in range(2):
                U[(2 + i_l) * 3 + j_l, col_offset + c] += U5[i_l * 2 + j_l, c]
        for j_l in range(2):
            for k_l in range(2):
                V[j_l * 5 + k_l, col_offset + c] += V5[j_l * 2 + k_l, c]
        for k_l in range(2):
            for i_l in range(1):
                W[k_l * 3 + (2 + i_l), col_offset + c] += W5[k_l * 1 + i_l, c]
    col_offset += 4
    
    # Block 6: 1x1x2 (Schoolbook)
    U6, V6, W6 = _schoolbook(1, 1, 2)
    for c in range(2):
        for i_l in range(1):
            for j_l in range(1):
                U[(2 + i_l) * 3 + (2 + j_l), col_offset + c] += U6[i_l * 1 + j_l, c]
        for j_l in range(1):
            for k_l in range(2):
                V[(2 + j_l) * 5 + k_l, col_offset + c] += V6[j_l * 2 + k_l, c]
        for k_l in range(2):
            for i_l in range(1):
                W[k_l * 3 + (2 + i_l), col_offset + c] += W6[k_l * 1 + i_l, c]
    col_offset += 2
    
    # Block 7: 1x2x3 (Schoolbook)
    U7, V7, W7 = _schoolbook(1, 2, 3)
    for c in range(6):
        for i_l in range(1):
            for j_l in range(2):
                U[(2 + i_l) * 3 + j_l, col_offset + c] += U7[i_l * 2 + j_l, c]
        for j_l in range(2):
            for k_l in range(3):
                V[j_l * 5 + (2 + k_l), col_offset + c] += V7[j_l * 3 + k_l, c]
        for k_l in range(3):
            for i_l in range(1):
                W[(2 + k_l) * 3 + (2 + i_l), col_offset + c] += W7[k_l * 1 + i_l, c]
    col_offset += 6
    
    # Block 8: 1x1x3 (Schoolbook)
    U8, V8, W8 = _schoolbook(1, 1, 3)
    for c in range(3):
        for i_l in range(1):
            for j_l in range(1):
                U[(2 + i_l) * 3 + (2 + j_l), col_offset + c] += U8[i_l * 1 + j_l, c]
        for j_l in range(1):
            for k_l in range(3):
                V[(2 + j_l) * 5 + (2 + k_l), col_offset + c] += V8[j_l * 3 + k_l, c]
        for k_l in range(3):
            for i_l in range(1):
                W[(2 + k_l) * 3 + (2 + i_l), col_offset + c] += W8[k_l * 1 + i_l, c]
    col_offset += 3
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
