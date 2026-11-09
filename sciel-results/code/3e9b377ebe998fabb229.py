def construct():
    n, m, p = 3, 5, 6
    rank = 36
    U = [[0]*rank for _ in range(n*m)]  # 15 x 36
    V = [[0]*rank for _ in range(m*p)]  # 30 x 36
    W = [[0]*rank for _ in range(p*n)]  # 18 x 36
    
    col = 0
    
    # Helper: add 2x2x1 block
    def add_2x2x1(i_off, j_off, k_off, c):
        nonlocal col
        # Term 1
        U[i_off*m + j_off][c] = 1
        U[i_off*m + j_off+1][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + i_off][c] = 1
        # Term 2
        U[(i_off+1)*m + j_off][c+1] = 1
        U[(i_off+1)*m + j_off+1][c+1] = 1
        V[j_off*p + k_off][c+1] = 1
        V[(j_off+1)*p + k_off][c+1] = 1
        W[k_off*n + (i_off+1)][c+1] = 1
        col += 2
    
    # Helper: add 2x1x6 block (rank 6)
    def add_2x1x6(i_off, j_off, k_off, c):
        nonlocal col
        for k in range(6):
            U[i_off*m + j_off][c+k] = 1
            U[(i_off+1)*m + j_off][c+k] = 1
            V[j_off*p + (k_off+k)][c+k] = 1
            W[(k_off+k)*n + i_off][c+k] = 1
            W[(k_off+k)*n + (i_off+1)][c+k] = 1
        col += 6
    
    # Helper: add 1x5x6 block (rank 6)
    def add_1x5x6(i_off, j_off, k_off, c):
        nonlocal col
        for k in range(6):
            for j in range(5):
                U[i_off*m + (j_off+j)][c+k] = 1
            V[(j_off+j)*p + (k_off+k)][c+k] = 1  # This is wrong, V depends on j
            # Actually, for 1x5x6: U[i*m+j, c] = 1 for all j, V[j*p+k, c] = 1 if c corresponds to k, W[k*n+i, c] = 1
            # But V[j*p+k, c] must be 1 only when the c-th term corresponds to output k.
            # So for term c (0-indexed within this block), it corresponds to k = c
            # V[j*p+k, c] should be 1 for the specific j? No, V[j*p+k, c] is per (j,k,c).
            # In the 1x5x6 decomposition: for output k, we use term c=k
            # U[i*m+j, c] = 1 for all j
            # V[j*p+k, c] = 1 for all j (when c==k)
            # W[k*n+i, c] = 1
            pass
    
    # This is getting messy. Let me just do it directly.
    
    # 2x2x6 block: i=0,1; j=0,1; k=0..5
    for k in range(6):
        add_2x2x1(0, 0, k, col)
    
    # 2x2x6 block: i=0,1; j=2,3; k=0..5
    for k in range(6):
        add_2x2x1(0, 2, k, col)
    
    # 2x1x6 block: i=0,1; j=4; k=0..5
    add_2x1x6(0, 4, 0, col)
    
    # 1x5x6 block: i=2; j=0..4; k=0..5
    # Rank 6: for each k, one term
    for k in range(6):
        for j in range(5):
            U[2*m + j][col] = 1
        V[j*p + k][col] = 1  # But j varies, V[j*p+k] is different for each j
        # We need V[j*p+k, col] = 1 for all j when this term is active
        for j in range(5):
            V[j*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
