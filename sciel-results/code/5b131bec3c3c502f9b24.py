def add_strassen_block(U, V, W, A_indices, B_indices, C_indices, base_col):
    # A_indices: [A00, A01, A10, A11] global indices (i*4+j)
    # B_indices: [B00, B01, B10, B11] global indices (j*4+k)
    # C_indices: [C00, C01, C10, C11] global indices (k*2+i)
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00=1, A11=1. V: B00=1, B11=1.
    U[A_indices[0]][base_col + 0] += 1
    U[A_indices[3]][base_col + 0] += 1
    V[B_indices[0]][base_col + 0] += 1
    V[B_indices[3]][base_col + 0] += 1
    # W: C00=1, C11=1
    W[C_indices[0]][base_col + 0] += 1
    W[C_indices[3]][base_col + 0] += 1
    
    # M2 = (A10 + A11)B00
    U[A_indices[2]][base_col + 1] += 1
    U[A_indices[3]][base_col + 1] += 1
    V[B_indices[0]][base_col + 1] += 1
    # W: C10=1
    W[C_indices[2]][base_col + 1] += 1
    
    # M3 = A00(B01 - B11)
    U[A_indices[0]][base_col + 2] += 1
    V[B_indices[1]][base_col + 2] += 1
    V[B_indices[3]][base_col + 2] -= 1
    # W: C01=1
    W[C_indices[1]][base_col + 2] += 1
    
    # M4 = A11(B10 - B00)
    U[A_indices[3]][base_col + 3] += 1
    V[B_indices[2]][base_col + 3] += 1
    V[B_indices[0]][base_col + 3] -= 1
    # W: C00=1, C10=1
    W[C_indices[0]][base_col + 3] += 1
    W[C_indices[2]][base_col + 3] += 1
    
    # M5 = (A00 + A01)B11
    U[A_indices[0]][base_col + 4] += 1
    U[A_indices[1]][base_col + 4] += 1
    V[B_indices[3]][base_col + 4] += 1
    # W: C00=-1, C01=1
    W[C_indices[0]][base_col + 4] -= 1
    W[C_indices[1]][base_col + 4] += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[A_indices[2]][base_col + 5] += 1
    U[A_indices[0]][base_col + 5] -= 1
    V[B_indices[0]][base_col + 5] += 1
    V[B_indices[1]][base_col + 5] += 1
    # W: C11=1
    W[C_indices[3]][base_col + 5] += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[A_indices[1]][base_col + 6] += 1
    U[A_indices[3]][base_col + 6] -= 1
    V[B_indices[2]][base_col + 6] += 1
    V[B_indices[3]][base_col + 6] += 1
    # W: C00=1
    W[C_indices[0]][base_col + 6] += 1
