<<<<<<< SEARCH
    # Initialize factor matrices as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Build the schoolbook decomposition
    # Each term c corresponds to a unique (i, j, k) triple
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Safety check (though execution is instantaneous)
    if time.time() - start_time > 110:
         pass # Just in case, but it won't happen
        
    return {"U": U, "V": V, "W": W}
=======
    # Initialize factor matrices as NumPy arrays of zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Build the schoolbook decomposition
    # Each term c corresponds to a unique (i, j, k) triple
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j, column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k, column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    # Safety check (though execution is instantaneous)
    if time.time() - start_time > 110:
         pass # Just in case, but it won't happen
        
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
