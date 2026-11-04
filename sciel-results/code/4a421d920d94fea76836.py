import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1
    ]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 0, 0, 1, 0],  # j=0, k=1
        [0, 0, 0, 1, 0, 0, 1],  # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1
    ]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0
        [0, 1, 0, 1, 0, 0, 0],  # k=0, i=1
        [0, 0, 1, 0, 1, 0, 0],  # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1
    ]

    column = 0
    # Split A into A1, A2 (2x2 each)
    # Split B into B1, B2 (2x8 each)
    # B1 split into B11, B12, B13, B14 (2x2 each)
    # B2 split into B21, B22, B23, B24 (2x2 each)
    
    for a_block in range(2):
        for b_block in range(8):
            # a_block 0 -> A1, 1 -> A2
            # b_block 0-3 -> B11-B14, 4-7 -> B21-B24
            
            # A index offset
            a_row_offset = a_block * 2
            
            # B index offset
            # If b_block < 4, it's part of B1 (rows 0-1)
            # Else part of B2 (rows 2-3)
            if b_block < 4:
                b_row_offset = 0
                b_col_offset = (b_block % 4) * 2
            else:
                b_row_offset = 2
                b_col_offset = ((b_block - 4) % 4) * 2
            
            # C index offset
            # C is 2x8.
            # Rows 0-1.
            # Cols determined by b_block.
            c_col_offset = (b_block % 4) * 2
            
            for k in range(7):
                # Add Strassen factors to U
                for i in range(2):
                    for j in range(2):
                        val = strassen_U[i * 2 + j][k]
                        if val != 0:
                            # U index: i_global * m + j_global
                            # i_global = a_block * 2 + i
                            # j_global = j
                            U[(a_block * 2 + i) * 4 + j][column] += val
                
                # Add Strassen factors to V
                for j in range(2):
                    for k_idx in range(2):
                        val = strassen_V[j * 2 + k_idx][k]
                        if val != 0:
                            # V index: j_global * p + k_global
                            # j_global = b_row_offset + j
                            # k_global = b_col_offset + k_idx
                            V[(b_row_offset + j) * 8 + (b_col_offset + k_idx)][column] += val
                
                # Add Strassen factors to W
                for k_idx in range(2):
                    for i in range(2):
                        val = strassen_W[k_idx * 2 + i][k]
                        if val != 0:
                            # W index: k_global * n + i_global
                            # k_global = c_col_offset + k_idx
                            # i_global = i
                            W[(c_col_offset + k_idx) * 2 + i][column] += val
                
                column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
