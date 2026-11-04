def construct():
    n, m, p = 3, 5, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Define the 6 Strassen blocks
    strassen_blocks = [
        (0, 1, 0, 1, 0, 1),  # i0,i1, j0,j1, k0,k1
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    
    # For each Strassen block, add 7 terms
    for (i0, i1, j0, j1, k0, k1) in strassen_blocks:
        # M1 = (A_{i0,j0} + A_{i1,j1}) * (B_{j0,k0} + B_{j1,k1})
        U[i0*m+j0][col] += 1
        U[i1*m+j1][col] += 1
        V[j0*p+k0][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        
        # M2 = (A_{i1,j0} + A_{i1,j1}) * B_{j0,k0}
        U[i1*m+j0][col] += 1
        U[i1*m+j1][col] += 1
        V[j0*p+k0][col] += 1
        W[k0*n+i1][col] += 1
        col += 1
        
        # M3 = A_{i0,j0} * (B_{j1,k0} - B_{j1,k1})
        U[i0*m+j0][col] += 1
        V[j1*p+k0][col] += 1
        V[j1*p+k1][col] += -1
        W[k0*n+i0][col] += 1
        col += 1
        
        # M4 = A_{i1,j1} * (B_{j0,k1} - B_{j0,k0})
        U[i1*m+j1][col] += 1
        V[j0*p+k1][col] += 1
        V[j0*p+k0][col] += -1
        W[k1*n+i1][col] += 1
        col += 1
        
        # M5 = (A_{i0,j0} + A_{i0,j1}) * B_{j1,k1}
        U[i0*m+j0][col] += 1
        U[i0*m+j1][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i0][col] += 1
        col += 1
        
        # M6 = (A_{i1,j0} - A_{i0,j0}) * (B_{j0,k0} + B_{j0,k1})
        U[i1*m+j0][col] += 1
        U[i0*m+j0][col] += -1
        V[j0*p+k0][col] += 1
        V[j0*p+k1][col] += 1
        W[k0*n+i1][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        
        # M7 = (A_{i0,j1} - A_{i1,j1}) * (B_{j1,k0} + B_{j1,k1})
        U[i0*m+j1][col] += 1
        U[i1*m+j1][col] += -1
        V[j1*p+k0][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] += 1
        W[k1*n+i0][col] += 1
        col += 1
    
    # Now add schoolbook terms for all (i,j,k) NOT in any Strassen block
    strassen_set = set()
    for (i0, i1, j0, j1, k0, k1) in strassen_blocks:
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    strassen_set.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_set:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
