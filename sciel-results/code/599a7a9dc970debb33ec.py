<<<<<<< SEARCH
    for q in range(3):
        for j in range(3):
            for s in range(3):
                r = q * 9 + j * 3 + s  # rank index 0..26
                
                # U[i, j, r] for i in {0,1}:
                # We want U to depend on i and the (q,j,s) group
                # Set U[i, j, r] = 1 for both i
                U_try[i * m + j][r] = 1 for i in range(n)  # This is wrong syntax
                
    # Let me redo this properly
    U_try = np.zeros((n * m, 27))
    V_try = np.zeros((m * p, 27))
    W_try = np.zeros((p * n, 27))
    
    for q in range(3):
        for j in range(3):
=======
    U_try = np.zeros((n * m, 27))
    V_try = np.zeros((m * p, 27))
    W_try = np.zeros((p * n, 27))
    
    for q in range(3):
        for j in range(3):
>>>>>>> REPLACE
