import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2
    # Blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A * B
    #
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    def add_strassen_block(base_col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows, a_cols: indices for the 2x2 submatrix of A
        # b_rows, b_cols: indices for the 2x2 submatrix of B
        # c_rows, c_cols: indices for the 2x2 submatrix of C
        
        # M1
        U[a_rows[0]*m + a_cols[0]][base_col+0] += 1
        U[a_rows[1]*m + a_cols[1]][base_col+0] += 1
        V[b_rows[0]*p + b_cols[0]][base_col+0] += 1
        V[b_rows[1]*p + b_cols[1]][base_col+0] += 1
        
        # M2
        U[a_rows[1]*m + a_cols[0]][base_col+1] += 1
        U[a_rows[1]*m + a_cols[1]][base_col+1] += 1
        V[b_rows[0]*p + b_cols[0]][base_col+1] += 1
        
        # M3
        U[a_rows[0]*m + a_cols[0]][base_col+2] += 1
        V[b_rows[0]*p + b_cols[1]][base_col+2] += 1
        V[b_rows[1]*p + b_cols[1]][base_col+2] -= 1
        
        # M4
        U[a_rows[1]*m + a_cols[1]][base_col+3] += 1
        V[b_rows[1]*p + b_cols[0]][base_col+3] += 1
        V[b_rows[0]*p + b_cols[0]][base_col+3] -= 1
        
        # M5
        U[a_rows[0]*m + a_cols[0]][base_col+4] += 1
        U[a_rows[0]*m + a_cols[1]][base_col+4] += 1
        V[b_rows[1]*p + b_cols[1]][base_col+4] += 1
        
        # M6
        U[a_rows[1]*m + a_cols[0]][base_col+5] += 1
        U[a_rows[0]*m + a_cols[0]][base_col+5] -= 1
        V[b_rows[0]*p + b_cols[0]][base_col+5] += 1
        V[b_rows[0]*p + b_cols[1]][base_col+5] += 1
        
        # M7
        U[a_rows[0]*m + a_cols[1]][base_col+6] += 1
        U[a_rows[1]*m + a_cols[1]][base_col+6] -= 1
        V[b_rows[1]*p + b_cols[0]][base_col+6] += 1
        V[b_rows[1]*p + b_cols[1]][base_col+6] += 1
        
        # Update W
        # C11
        w_idx = c_cols[0]*n + c_rows[0]
        W[w_idx][base_col+0] += 1 # M1
        W[w_idx][base_col+3] += 1 # M4
        W[w_idx][base_col+4] -= 1 # -M5
        W[w_idx][base_col+6] += 1 # M7
        
        # C12
        w_idx = c_cols[1]*n + c_rows[0]
        W[w_idx][base_col+2] += 1 # M3
        W[w_idx][base_col+4] += 1 # M5
        
        # C21
        w_idx = c_cols[0]*n + c_rows[1]
        W[w_idx][base_col+1] += 1 # M2
        W[w_idx][base_col+3] += 1 # M4
        
        # C22
        w_idx = c_cols[1]*n + c_rows[1]
        W[w_idx][base_col+0] += 1 # M1
        W[w_idx][base_col+1] -= 1 # -M2
        W[w_idx][base_col+2] += 1 # M3
        W[w_idx][base_col+5] += 1 # M6

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(7, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(14, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(21, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])

    return {"U": U, "V": V, "W": W}
