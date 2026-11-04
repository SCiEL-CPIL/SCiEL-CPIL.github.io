import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base Strassen decomposition for 2x2x2
    # U2, V2, W2 are 4x7 matrices
    # Rows: 00, 01, 10, 11
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, -1, 0, 1, 0, 0, -1],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, -1, 0, 1, 1, 0]
    ])
    
    # Block indices: 0=00, 1=01, 2=10, 3=11
    # Rows in 4x4 matrix for each block
    # A (n x m = 4 x 4)
    # Row index i*m + j
    # Block 00: i in {0,1}, j in {0,1} -> rows 0,1,4,5
    # Block 01: i in {0,1}, j in {2,3} -> rows 2,3,6,7
    # Block 10: i in {2,3}, j in {0,1} -> rows 8,9,12,13
    # Block 11: i in {2,3}, j in {2,3} -> rows 10,11,14,15
    A_blocks = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # B (m x p_block = 4 x 4)
    # Row index j*p_block + k
    # Block 00: j in {0,1}, k in {0,1} -> rows 0,1,4,5
    # Block 01: j in {0,1}, k in {2,3} -> rows 2,3,6,7
    # Block 10: j in {2,3}, k in {0,1} -> rows 8,9,12,13
    # Block 11: j in {2,3}, k in {2,3} -> rows 10,11,14,15
    B_blocks = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # C (p_block x n = 4 x 4)
    # Row index k*n + i
    # Block 00: k in {0,1}, i in {0,1} -> rows 0,1,4,5
    # Block 01: k in {0,1}, i in {2,3} -> rows 2,3,6,7
    # Block 10: k in {2,3}, i in {0,1} -> rows 8,9,12,13
    # Block 11: k in {2,3}, i in {2,3} -> rows 10,11,14,15
    C_blocks = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # Super-term block coefficients
    # L_A[l][a]
    L_A = [
        [1, 0, 0, 1], # P1: A00, A11
        [0, 0, 1, 1], # P2: A10, A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 1, 0, 0], # P5: A00, A01
        [-1, 0, 1, 0], # P6: A00(-1), A10
        [0, 1, 0, -1]  # P7: A01, A11(-1)
    ]
    L_B = [
        [1, 0, 0, 1], # P1: B00, B11
        [1, 0, 0, 0], # P2: B00
        [0, 1, 0, -1], # P3: B01, B11(-1)
        [0, 0, 1, -1], # P4: B10, B00(-1)
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00, B01
        [0, 0, 1, 1]  # P7: B10, B11
    ]
    
    # Construct 4x4x4 Strassen decomposition
    # Rank is 49
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    col = 0
    for l in range(7):
        for r in range(7):
            # Add contributions from U2, V2, W2
            for a in range(4):
                coeff_a = L_A[l][a]
                if coeff_a != 0:
                    val = coeff_a * U2[a, r]
                    if val != 0:
                        for row in A_blocks[a]:
                            U_44[row, col] += val
            for b in range(4):
                coeff_b = L_B[l][b]
                if coeff_b != 0:
                    val = coeff_b * V2[b, r]
                    if val != 0:
                        for row in B_blocks[b]:
                            V_44[row, col] += val
            for c in range(4):
                coeff_c = 1 # W2 already contains the coefficients for C blocks
                val = W2[c, r]
                if val != 0:
                    for row in C_blocks[c]:
                        W_44[row, col] += val
            col += 1
            
    # Now assemble 4x4x9
    # Split 9 cols into 4, 4, 1
    # U is 16 x Rank
    # V is 36 x Rank
    # W is 36 x Rank
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1: cols 0-3
    U[:, 0:49] = U_44
    V[0:16, 0:49] = V_44
    W[0:16, 0:49] = W_44
    
    # Block 2: cols 4-7
    U[:, 49:98] = U_44
    V[16:32, 49:98] = V_44
    W[16:32, 49:98] = W_44
    
    # Block 3: col 8
    # Schoolbook for 4x4x1
    # C[i, 8] = sum_j A[i, j] B[j, 8]
    # Rank 16
    col_offset = 98
    for i in range(4):
        for j in range(4):
            # Term A[i,j] * B[j,8]
            U[i*4 + j, col_offset] = 1
            V[j*9 + 8, col_offset] = 1
            W[8*4 + i, col_offset] = 1
            col_offset += 1
            
    # Convert to lists of lists as required by interface?
    # Interface: {'U', 'V', 'W'}: factor matrices of shapes ...
    # The seed program returns lists of lists.
    # Let's check if numpy arrays are accepted or if we need lists.
    # "integer/half-integer entries"
    # Usually, returning numpy arrays is fine if the evaluator handles it, 
    # but the seed returns lists. To be safe, I'll return lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
