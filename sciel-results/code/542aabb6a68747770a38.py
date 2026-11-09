def add_strassen_block(U, V, W, A_start_i, A_start_j, B_start_j, B_start_k, C_start_i, C_start_k, col_start):
    # A is 2x2 starting at row A_start_i, col A_start_j
    # B is 2x2 starting at row B_start_j, col B_start_k
    # C is 2x2 starting at row C_start_i, col C_start_k
    # col_start is the column index in U,V,W to write to
    
    # Strassen multipliers M1..M7
    # U corresponds to A, V to B, W to C
    
    # M1 = (A11+A22)(B11+B22)
    U[A_start_i * 4 + A_start_j][col_start] += 1
    U[(A_start_i+1) * 4 + (A_start_j+1)][col_start] += 1
    V[B_start_j * 4 + B_start_k][col_start] += 1
    V[(B_start_j+1) * 4 + (B_start_k+1)][col_start] += 1
    W[C_start_k * 2 + C_start_i][col_start] += 1
    W[(C_start_k+1) * 2 + (C_start_i+1)][col_start] += 1
    # Wait, C is 2x2. C11, C12, C21, C22.
    # Z11 = M1 + M4 - M5 - M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # The W matrix in the decomposition sums the products.
    # So for M1, it contributes to Z11, Z22.
    # We must add the coefficients to W.
    
    # M1
    W[C_start_k * 2 + C_start_i][col_start] = 1
    W[(C_start_k+1) * 2 + (C_start_i+1)][col_start] = 1
    
    # M2 = (A11+A22)B21
    U[A_start_i * 4 + A_start_j][col_start+1] += 1
    U[(A_start_i+1) * 4 + (A_start_j+1)][col_start+1] += 1
    V[(B_start_j+1) * 4 + B_start_k][col_start+1] += 1
    W[C_start_k * 2 + (C_start_i+1)][col_start+1] = 1 # Z21
    W[(C_start_k+1) * 2 + (C_start_i+1)][col_start+1] = -1 # Z22
    
    # M3 = A11(B12-B22)
    U[A_start_i * 4 + A_start_j][col_start+2] += 1
    V[B_start_j * 4 + (B_start_k+1)][col_start+2] += 1
    V[(B_start_j+1) * 4 + (B_start_k+1)][col_start+2] -= 1
    W[C_start_k * 2 + (C_start_i+1)][col_start+2] = 1 # Z12
    W[(C_start_k+1) * 2 + (C_start_i+1)][col_start+2] = 1 # Z22
    
    # M4 = A22(B11+B21)
    U[(A_start_i+1) * 4 + (A_start_j+1)][col_start+3] += 1
    V[B_start_j * 4 + B_start_k][col_start+3] += 1
    V[(B_start_j+1) * 4 + B_start_k][col_start+3] += 1
    W[C_start_k * 2 + C_start_i][col_start+3] = 1 # Z11
    W[C_start_k * 2 + (C_start_i+1)][col_start+3] = 1 # Z21
    
    # M5 = (A12+A22)B11
    U[A_start_i * 4 + (A_start_j+1)][col_start+4] += 1
    U[(A_start_i+1) * 4 + (A_start_j+1)][col_start+4] += 1
    V[B_start_j * 4 + B_start_k][col_start+4] += 1
    W[C_start_k * 2 + C_start_i][col_start+4] = -1 # Z11
    W[C_start_k * 2 + (C_start_i+1)][col_start+4] = 1 # Z12
    
    # M6 = (A11-A12)(B21+B22)
    U[A_start_i * 4 + A_start_j][col_start+5] += 1
    U[A_start_i * 4 + (A_start_j+1)][col_start+5] -= 1
    V[(B_start_j+1) * 4 + B_start_k][col_start+5] += 1
    V[(B_start_j+1) * 4 + (B_start_k+1)][col_start+5] += 1
    W[(C_start_k+1) * 2 + (C_start_i+1)][col_start+5] = 1 # Z22
    
    # M7 = (A21-A11)(B11+B12)
    U[(A_start_i+1) * 4 + A_start_j][col_start+6] += 1
    U[A_start_i * 4 + A_start_j][col_start+6] -= 1
    V[B_start_j * 4 + B_start_k][col_start+6] += 1
    V[B_start_j * 4 + (B_start_k+1)][col_start+6] += 1
    W[C_start_k * 2 + C_start_i][col_start+6] = -1 # Z11
    
    return
