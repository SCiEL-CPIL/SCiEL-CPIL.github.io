def construct():
    n, m, p = 3, 3, 5
    # Part 1: 3x3x3 schoolbook (27 terms)
    # Part 2: 3x3x2 optimized (6 terms)
    
    rank = 27 + 6
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Part 1
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    # Part 2
    # k=3 (index 3 in p)
    for r in range(m): # r corresponds to j
        U[0*m + r][col] = 1 # Wait, U depends on i, j.
        # In the optimized 3x3x1, U_{i,j,r} = delta_{j,r}.
        # So for a fixed r, U_{i,j,r} is 1 if j=r, else 0.
        # This means for each r, we need to set U[i*m + r][col] = 1 for all i?
        # No, U is a matrix. U[i*m + j, col].
        # If j=r, U[i*m + r, col] = 1 for all i.
        # If j!=r, U[i*m + j, col] = 0.
        
        # Let's re-verify the decomposition for 3x3x1.
        # C_i = sum_j A_{i,j} B_{j,k}.
        # Term r (for j=r):
        # U_{i,j,r} = 1 if j=r else 0.
        # V_{j,k,r} = 1 if j=r else 0.
        # W_{k,i,r} = 1.
        # Sum over r: sum_r (1 if j=r else 0) * A_{i,j} * (1 if j=r else 0) * B_{j,k} * 1
        # = sum_r delta_{j,r} A_{i,j} delta_{j,r} B_{j,k}
        # = A_{i,j} B_{j,k} (since sum_r delta_{j,r} = 1 for valid j).
        
        # So for each r (0 to 2):
        # Set U[i*m + r, col] = 1 for all i in 0..2.
        # Set V[r*p + k, col] = 1.
        # Set W[k*n + i, col] = 1 for all i in 0..2.
        
        for i in range(n):
            U[i*m + r][col] = 1
            W[k*n + i][col] = 1
        V[r*p + k][col] = 1
        col += 1
        
    # Do this for k=3 and k=4
    for k in [3, 4]:
        for r in range(m):
            for i in range(n):
                U[i*m + r][col] = 1
                W[k*n + i][col] = 1
            V[r*p + k][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
