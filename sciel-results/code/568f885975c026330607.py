def _laderman_factors():
    """
    Returns U, V, W for 3x3 matrix multiplication using schoolbook (rank 27).
    U: (9, 27), V: (9, 27), W: (9, 27)
    Indexing: U[i*3+j, r] for A[i,j], V[j*3+k, r] for B[j,k], W[k*3+i, r] for C[i,k]
    """
    U = [[0]*27 for _ in range(9)]
    V = [[0]*27 for _ in range(9)]
    W = [[0]*27 for _ in range(9)]
    
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][r] = 1
                V[j*3+k][r] = 1
                W[k*3+i][r] = 1
                r += 1
                
    return U, V, W
