<<<<<<< SEARCH
    col = 0
    for b in range(5):
        for r in range(7):
            # U: same A-coefficients for every block
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            # V: B-coefficients mapped to global indices for block b
            # e = B[0, 2b]   -> index 0*p + 2b     = 2b
            # f = B[0, 2b+1] -> index 0*p + 2b+1   = 2b+1
            # g = B[1, 2b]   -> index 1*p + 2b     = 10 + 2b
            # h = B[1, 2b+1] -> index 1*p + 2b+1   = 10 + 2b+1
            V[2*b][col]       = strassen_B[r][0]
            V[2*b+1][col]     = strassen_B[r][1]
            V[10 + 2*b][col]  = strassen_B[r][2]
            V[10 + 2*b+1][col]= strassen_B[r][3]
            
            # W: C-coefficients mapped to global indices (trace convention k*n+i)
            # C00: i=0, k=2b     -> 2b*2 + 0 = 4b
            # C01: i=0, k=2b+1   -> (2b+1)*2 + 0 = 4b+2
            # C10: i=1, k=2b     -> 2b*2 + 1 = 4b+1
            # C11: i=1, k=2b+1   -> (2b+1)*2 + 1 = 4b+3
            W[4*b][col]    = w_contrib[r][0]   # C00
            W[4*b+2][col]  = w_contrib[r][1]   # C01
            W[4*b+1][col]  = w_contrib[r][2]   # C10
            W[4*b+3][col]  = w_contrib[r][3]   # C11
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
    col = 0
    for b in range(5):
        for r in range(7):
            # U: same A-coefficients for every block
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            # V: B-coefficients mapped to global indices for block b
            # e = B[0, 2b]   -> index 0*p + 2b     = 2b
            # f = B[0, 2b+1] -> index 0*p + 2b+1   = 2b+1
            # g = B[1, 2b]   -> index 1*p + 2b     = 10 + 2b
            # h = B[1, 2b+1] -> index 1*p + 2b+1   = 10 + 2b+1
            V[2*b][col]       = strassen_B[r][0]
            V[2*b+1][col]     = strassen_B[r][1]
            V[10 + 2*b][col]  = strassen_B[r][2]
            V[10 + 2*b+1][col]= strassen_B[r][3]
            
            # W: C-coefficients mapped to global indices (trace convention k*n+i)
            # C00: i=0, k=2b     -> 2b*2 + 0 = 4b
            # C01: i=0, k=2b+1   -> (2b+1)*2 + 0 = 4b+2
            # C10: i=1, k=2b     -> 2b*2 + 1 = 4b+1
            # C11: i=1, k=2b+1   -> (2b+1)*2 + 1 = 4b+3
            W[4*b][col]    = w_contrib[r][0]   # C00
            W[4*b+2][col]  = w_contrib[r][1]   # C01
            W[4*b+1][col]  = w_contrib[r][2]   # C10
            W[4*b+3][col]  = w_contrib[r][3]   # C11
            
            col += 1
    
    # Verify the decomposition before returning
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
