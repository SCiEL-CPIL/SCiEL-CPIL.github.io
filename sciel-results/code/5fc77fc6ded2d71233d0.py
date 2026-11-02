"""Rank-28 exact tensor decomposition for 2x4 * 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen's 2x2 algorithm (rank 7) for each of the 4 block products.
    # A is 2x4, B is 4x4.
    # Split A into A_L (2x2) and A_R (2x2).
    # Split B into B_00 (2x2), B_01 (2x2), B_10 (2x2), B_11 (2x2).
    # C = A * B = [A_L B_00 + A_R B_10, A_L B_01 + A_R B_11]
    # Each block product is 2x2 * 2x2 -> 2x2, which can be done with 7 mults.
    # Total rank = 4 * 7 = 28.

    n, m, p = 2, 4, 4
    rank = 28

    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)
    # W has shape (p*n, rank) = (8, 28)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each block product, we have 7 columns.
    # Block indices in A (rows 0,1 for left; 2,3 for right in the 8-dim space):
    # A_L: indices 0,1,2,3 (A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3)
    # A_R: indices 4,5,6,7 (A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7)
    
    # B indices (16-dim space):
    # B_00: 0,1,2,3 (B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3)
    # B_01: 4,5,6,7 (B[0,2]=4, B[0,3]=5, B[1,2]=6, B[1,3]=7)
    # B_10: 8,9,10,11 (B[2,0]=8, B[2,1]=9, B[3,0]=10, B[3,1]=11)
    # B_11: 12,13,14,15 (B[2,2]=12, B[2,3]=13, B[3,2]=14, B[3,3]=15)

    # C indices (8-dim space):
    # C_L: 0,1,2,3 (C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3)
    # C_R: 4,5,6,7 (C[0,2]=4, C[0,3]=5, C[1,2]=6, C[1,3]=7)

    # Strassen U coefficients (for 2x2 A):
    # M1: A00+1, A11+1 -> [1,0,0,1]
    # M2: A10+1, A11+1 -> [0,0,1,1]
    # M3: A00+1 -> [1,0,0,0]
    # M4: A11+1 -> [0,0,0,1]
    # M5: A00+1, A01+1 -> [1,1,0,0]
    # M6: A10+1, A00-1 -> [-1,0,1,0]
    # M7: A01+1, A11-1 -> [0,1,-1,0]

    # Strassen V coefficients (for 2x2 B):
    # M1: B00+1, B11+1 -> [1,0,0,1]
    # M2: B00+1 -> [1,0,0,0]
    # M3: B10+1, B11-1 -> [0,-1,1,0]
    # M4: B01+1, B00-1 -> [-1,1,0,0]
    # M5: B11+1 -> [0,0,0,1]
    # M6: B00+1, B01+1 -> [1,1,0,0]
    # M7: B10+1, B11+1 -> [0,0,1,1]

    # Strassen W coefficients (for 2x2 C):
    # C00: M1+1, M4-1, M5+1, M7+1
    # C01: M3+1, M5+1
    # C10: M2+1, M4+1
    # C11: M1+1, M2-1, M3+1, M6+1

    U_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, -1, 0]  # M7
    ]

    V_strassen = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, -1, 1, 0], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]

    W_strassen = [
        [1, 0, 0, -1, 1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]

    # Define the 4 block products:
    # (A_offset, B_offset, C_offset)
    # A_offset is the row offset in A (0 or 4)
    # B_offset is the row offset in B (0, 4, 8, or 12)
    # C_offset is the row offset in C (0 or 4)
    blocks = [
        (0, 0, 0),    # A_L * B_00 -> C_L
        (4, 8, 0),    # A_R * B_10 -> C_L
        (0, 4, 4),    # A_L * B_01 -> C_R
        (4, 12, 4)    # A_R * B_11 -> C_R
    ]

    col = 0
    for a_off, b_off, c_off in blocks:
        for k in range(7):
            # U coefficients
            for i in range(4):
                val = U_strassen[k][i]
                if val != 0:
                    U[a_off + i][col] += val

            # V coefficients
            for j in range(4):
                val = V_strassen[k][j]
                if val != 0:
                    V[b_off + j][col] += val

            # W coefficients
            for c in range(4):
                val = W_strassen[c][k]
                if val != 0:
                    W[c_off + c][col] += val

            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
