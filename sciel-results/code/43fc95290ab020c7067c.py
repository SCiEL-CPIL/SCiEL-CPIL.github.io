    # Add A2 B2 part (rank 1)
    # U[i, 2] = 1 for i=0,1
    # V[2, k] = 1 for k=0..8
    # W[k, i] = 1 for all k,i
    
    for i in range(2):
        U_mat[i*m + 2][col] = 1
    for k in range(9):
        V_mat[2*p + k][col] = 1
    for k in range(9):
        for i in range(2):
            W_mat[k*n + i][col] = 1
    col += 1
