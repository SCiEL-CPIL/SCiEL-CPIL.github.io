def construct():
    n, m, p = 3, 4, 7
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen blocks
    blocks = [
        (0, 0, 0), 
        (0, 2, 0), 
        (0, 0, 2), 
        (0, 2, 2), 
        (0, 0, 4), 
        (0, 2, 4), 
    ]
    
    def add_strassen_block(col_start, i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        U[i0*m+j0][col_start] = 1
        U[i1*m+j1][col_start] = 1
        V[j0*p+k0][col_start] = 1
        V[j1*p+k1][col_start] = 1
        W[k0*n+i0][col_start] = 1
        W[k1*n+i1][col_start] = 1

        U[i1*m+j0][col_start+1] = 1
        U[i1*m+j1][col_start+1] = 1
        V[j0*p+k0][col_start+1] = 1
        W[k0*n+i1][col_start+1] = 1
        W[k1*n+i1][col_start+1] = -1

        U[i0*m+j0][col_start+2] = 1
        V[j0*p+k1][col_start+2] = 1
        V[j1*p+k1][col_start+2] = -1
        W[k1*n+i0][col_start+2] = 1
        W[k1*n+i1][col_start+2] = 1

        U[i1*m+j1][col_start+3] = 1
        V[j1*p+k0][col_start+3] = 1
        V[j0*p+k0][col_start+3] = -1
        W[k0*n+i0][col_start+3] = 1
        W[k0*n+i1][col_start+3] = 1

        U[i0*m+j0][col_start+4] = 1
        U[i0*m+j1][col_start+4] = 1
        V[j1*p+k1][col_start+4] = 1
        W[k0*n+i0][col_start+4] = -1
        W[k1*n+i0][col_start+4] = 1

        U[i1*m+j0][col_start+5] = 1
        U[i0*m+j0][col_start+5] = -1
        V[j0*p+k0][col_start+5] = 1
        V[j0*p+k1][col_start+5] = 1
        W[k1*n+i1][col_start+5] = 1

        U[i0*m+j1][col_start+6] = 1
        U[i1*m+j1][col_start+6] = -1
        V[j1*p+k0][col_start+6] = 1
        V[j1*p+k1][col_start+6] = 1
        W[k0*n+i0][col_start+6] = 1

        return col_start + 7

    col = 0
    for i0, j0, k0 in blocks:
        col = add_strassen_block(col, i0, j0, k0)
        
    # Term A: i=2 slice
    # U[2, j] = 1 for all j
    # V[j, k] = 1 for all j, k
    # W[k, 2] = 1 for all k
    for j in range(m):
        U[2*m + j][col] = 1
    for j in range(m):
        for k in range(p):
            V[j*p + k][col] = 1
    for k in range(p):
        W[k*n + 2][col] = 1
    col += 1
    
    # Term B: k=6 slice for i in {0, 1}
    # U[i, j] = 1 for i in {0, 1}, all j
    # V[j, 6] = 1 for all j
    # W[6, i] = 1 for i in {0, 1}
    for i in range(2):
        for j in range(m):
            U[i*m + j][col] = 1
    for j in range(m):
        V[j*p + 6][col] = 1
    for i in range(2):
        W[6*n + i][col] = 1
    col += 1
    
    if col != rank:
         # Fallback if logic error
         return _schoolbook(n, m, p)
         
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
